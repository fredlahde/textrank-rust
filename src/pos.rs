use senna::senna::{Senna, SennaParseOptions};
use senna::sennapath::SENNA_PATH;

pub(crate) struct Tagger {
    senna: Senna,
}
impl Tagger {
    pub(crate) fn new() -> Tagger {
        let mut senna = Senna::new(SENNA_PATH.to_owned());
        Tagger { senna }
    }

    pub(crate) fn tag(&mut self, tokens: &[String]) -> Vec<String> {
        let mut joined = String::default();
        for s in tokens {
            joined.push_str(s);
            joined.push_str(" ");
        }
        let sentence = self.senna.parse(
            &joined,
            SennaParseOptions {
                pos: true,
                psg: false,
            },
        );
        sentence
            .get_words()
            .iter()
            .map(|w| w.get_pos().to_str().to_owned())
            .collect::<Vec<_>>()
    }
}
