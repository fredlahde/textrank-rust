use std::io::Read;
pub(crate) fn get_stoplist() -> std::io::Result<Vec<String>> {
    let path = "fox-stoplist.txt";
    let mut text = String::new();
    let mut f = std::fs::File::open(path)?;
    f.read_to_string(&mut text)?;
    Ok(text.split('\n').map(|s| s.to_owned()).collect())
}
