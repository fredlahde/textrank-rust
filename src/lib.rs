extern crate ndarray;
extern crate senna;
extern crate serde;
extern crate serde_derive;
extern crate serde_json;
mod pos;
mod stoplist;
mod util;
use std::collections::HashMap;
use util::cmp_f64;

/// damping coefficient
const D: f64 = 0.85;
/// convergence threshold
const MIN_DIFF: f64 = 1e-5;
/// iteration steps
const STEPS: usize = 20;
const WINDOW_SIZE: usize = 2;

#[derive(Eq, PartialEq, Hash, Debug)]
pub struct Token {
    pub term: String,
    pub offset_begin: usize,
    pub pos: Option<String>,
}

impl Clone for Token {
    fn clone(&self) -> Self {
        Token {
            term: self.term.clone(),
            offset_begin: self.offset_begin,
            pos: match &self.pos {
                Some(pos) => Some(pos.clone()),
                None => None,
            },
        }
    }
}

impl Token {
    pub fn offset(&self) -> (usize, usize) {
        (self.offset_begin, self.offset_begin + self.term.len())
    }
}

pub type Sentence = Vec<Token>;
pub type Vocab = HashMap<Token, usize>;

fn get_vocab(sentences: &[Sentence]) -> Vocab {
    let mut ret = HashMap::new();
    let mut idx = 0;
    for sentence in sentences {
        for word in sentence {
            if !ret.contains_key(word) {
                ret.insert(word.to_owned(), idx);
            }
            idx += 1;
        }
    }
    ret
}

fn get_token_pairs(window_size: usize, sentences: &[Sentence]) -> Vec<(Token, Token)> {
    let mut ret = Vec::new();
    for sentence in sentences {
        for (i, word) in sentence.iter().enumerate() {
            for j in i + 1..i + window_size {
                if j >= sentence.len() {
                    break;
                }
                let pair = (word.clone(), sentence[j].clone());
                if !ret.contains(&pair) {
                    ret.push(pair);
                }
            }
        }
    }
    ret
}

fn get_matrix(
    vocab: &Vocab,
    token_pairs: Vec<(Token, Token)>,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    // build matrix
    let vocab_size = vocab.len();
    let mut g = ndarray::Array::zeros((vocab_size, vocab_size));
    for (w1, w2) in token_pairs.iter() {
        let (i, j) = (vocab[&w1], vocab[&w2]);
        if i >= vocab_size || j >= vocab_size {
            break;
        }
        g[[i, j]] = 1f64
    }

    // get symmetric matrix
    let added = &g + &g.t();
    let diag = &g.diag();
    let diag = ndarray::Array::from_diag(&diag);
    let sym = added - diag;

    // normalise matrix
    let mut max = std::f64::MIN;
    let mut min = std::f64::MAX;
    for n in sym.iter() {
        let n: f64 = *n;
        if n > max {
            max = n;
        }
        if n < min {
            min = n;
        }
    }
    (sym - min) / (max - min)
}

const VALID_POS: &[&str] = &["NN", "NNP", "NNS"];
const PUNCTUATION: &[&str] = &[",", ".", "!", ":"];

/// implementation of TextRank
pub fn analyze(doc: Vec<Sentence>) -> std::io::Result<Vec<(Token, f64)>> {
    // TODO pos, window_size, lower, stopwords
    let stopwords = stoplist::get_stoplist().unwrap();
    let mut new_doc = vec![];
    let mut tagger = pos::Tagger::new();
    for sent in doc {
        let mut new_sent = vec![];
        let tags = tagger.tag(
            sent.iter()
                .map(|t| t.term.clone())
                .collect::<Vec<_>>()
                .as_slice(),
        );
        for (i, tok) in sent.iter().enumerate() {
            if stopwords.contains(&tok.term)
                || PUNCTUATION.contains(&tok.term.as_str())
                || !VALID_POS.contains(&tags[i].as_str())
            {
                continue;
            }
            new_sent.push(tok.clone());
        }
        new_doc.push(new_sent);
    }
    let vocab = get_vocab(&new_doc);
    let token_pairs = get_token_pairs(WINDOW_SIZE, &new_doc);
    let vocab_matrix = get_matrix(&vocab, token_pairs);
    let mut page_rank = ndarray::Array::<f64, _>::ones(vocab.len());
    let mut previous_page_rank = 0f64;

    for _ in 1..=STEPS {
        page_rank = (1f64 - D) + D * vocab_matrix.dot(&page_rank);
        if (previous_page_rank - page_rank.sum()).abs() < MIN_DIFF {
            break;
        }
        previous_page_rank = page_rank.sum();
    }

    let mut node_weight_vec: Vec<(Token, f64)> = vec![];
    for (i, word) in vocab.iter().enumerate() {
        node_weight_vec.push((word.0.clone(), page_rank[i]));
    }

    node_weight_vec.sort_by(|a, b| cmp_f64(a.1, b.1));
    Ok(node_weight_vec)
}

#[cfg(test)]
mod test {}
