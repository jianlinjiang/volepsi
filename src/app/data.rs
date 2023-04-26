use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Data {
    pub id: u64,
    pub register_date: String,
    pub age: u16,
}

#[derive(Deserialize, Debug)]
pub struct Id {
    pub id: u64,
}

pub fn encode_data(year: u16, month: u16, day: u16, age: u16) -> u64 {
    let mut res = (year as u64) << 48;
    res += (month as u64) << 32;
    res = res + ((day as u64) << 16);
    res = res + age as u64;
    res
}

pub fn decode_data(a: u64) -> (u16, u16, u16, u16) {
    let x: u64 = 0x000000000000ffff;
    let y = (a >> 48 & x) as u16;
    let m = (a >> 32 & x) as u16;
    let d = (a >> 16 & x) as u16;
    let age = (a & x) as u16;
    (y, m, d, age)
}

#[cfg(test)]
mod tests {
    use super::*;
    use csv::Reader;
    use rand::{thread_rng, RngCore};
    #[test]
    fn encode_test() {
        let mut rng = thread_rng();
        for _i in 0..1000000 {
            let mut data: [u16; 4] = [0; 4];
            data.iter_mut().for_each(|x| {
                *x = rng.next_u32() as u16;
            });

            let x = encode_data(data[0], data[1], data[2], data[3]);
            assert_eq!(decode_data(x), (data[0], data[1], data[2], data[3]));
        }
    }

    #[test]
    fn process_data() {
        let mut reader = Reader::from_path("data/B_PIR_DATA.csv").unwrap();
        let mut iter = reader.deserialize::<Data>();
        let size: usize = 1000000;
        let mut key: Vec<u64> = Vec::with_capacity(size);
        let mut value: Vec<u64> = Vec::with_capacity(size);
        while let Some(result) = iter.next() {
            let record = result.unwrap();
            let date: Vec<u16> = record
                .register_date
                .split("-")
                .map(|x| x.parse().unwrap())
                .collect();
            key.push(record.id);
            value.push(encode_data(date[0], date[1], date[2], record.age));
        }
        assert_eq!(key.len(), size);
        assert_eq!(value.len(), size);
    }
}
