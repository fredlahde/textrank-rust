extern crate ndarray;
use std::collections::HashMap;
/// damping coefficient
const D: f64 = 0.85;
/// convergence threshold
const MIN_DIFF: f64 = 1e-5;
/// iteration steps
const STEPS: usize = 10;
const WINDOW_SIZE: usize = 4;

#[derive(Eq, PartialEq, Hash)]
pub struct Token {
    term: String,
    offset_begin: usize,
}

impl Clone for Token {
    fn clone(&self) -> Self {
        Token {
            term: self.term.clone(),
            offset_begin: self.offset_begin,
        }
    }
}

pub type Sentence = Vec<Token>;
pub type Vocab = HashMap<Token, usize>;

impl Token {
    pub fn offset(&self) -> (usize, usize) {
        (self.offset_begin, self.offset_begin + self.term.len())
    }
}

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
    for (w1, w2) in token_pairs {
        let (i, j) = (vocab[&w1], vocab[&w2]);
        g[[i, j]] = 1f64
    }

    // get symmetric matrix
    let added = &g + &g.t();
    let diag = &g.diag();
    let diag = ndarray::Array::from_diag(&diag);
    let sym = added - diag;

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

/// implementation of TextRank
pub fn analyze(doc: Vec<Sentence>) {
    // TODO pos, window_size, lower, stopwords
    let vocab = get_vocab(&doc);
    let token_pairs = get_token_pairs(WINDOW_SIZE, &doc);
    let vocab_matrix = get_matrix(&vocab, token_pairs);
    let mut page_rank = ndarray::Array::<f64, _>::ones(vocab.len());
    let mut previous_page_rank = 0f64;

    for _ in 1..STEPS {
        page_rank = (1f64 - D) + D * vocab_matrix.dot(&page_rank);
        if (previous_page_rank - page_rank.sum()).abs() < MIN_DIFF {
            break;
        }
        previous_page_rank = page_rank.sum();
    }

    let mut node_weight: HashMap<_, f64> = HashMap::new();
    for (i, word) in vocab.iter().enumerate() {
        node_weight.insert(word, page_rank[[i]]);
    }
}
