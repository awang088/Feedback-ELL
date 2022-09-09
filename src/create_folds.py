import pandas as pd
import argparse
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from text_unidecode import unidecode
import codecs
from typing import Tuple

def get_args():
    parser = argparse.ArgumentParser(description="Fold Creation Args")
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)

    return text

def main():
    args = get_args()
    train = pd.read_csv('../input/fb3/train.csv')
    train['full_text'] = train['full_text'].apply(lambda x: resolve_encodings_and_normalize(x))
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

    mskf = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for n, (train_index, val_index) in enumerate(mskf.split(train, train[target_cols])):
        train.loc[val_index, 'fold'] = int(n)
    
    train.to_csv('../input/fb3/train_folds.csv', index=False)

if __name__ == '__main__':
    main()