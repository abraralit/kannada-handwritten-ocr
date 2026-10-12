"""Kannada Unicode table, vocabulary validation and CTC label decoding."""
from collections import defaultdict

import torch

# Kannada block of the Unicode standard (U+0C80 - U+0CFF)
KANNADA_UNICODE_CONSORTIUM = {
    'ಀ': '0C80', 'ಁ': '0C81', 'ಂ': '0C82', 'ಃ': '0C83', '಄': '0C84',
    'ಅ': '0C85', 'ಆ': '0C86', 'ಇ': '0C87', 'ಈ': '0C88', 'ಉ': '0C89',
    'ಊ': '0C8A', 'ಋ': '0C8B', 'ಌ': '0C8C', 'ಎ': '0C8E', 'ಏ': '0C8F',
    'ಐ': '0C90', 'ಒ': '0C92', 'ಓ': '0C93', 'ಔ': '0C94',
    'ಕ': '0C95', 'ಖ': '0C96', 'ಗ': '0C97', 'ಘ': '0C98', 'ಙ': '0C99',
    'ಚ': '0C9A', 'ಛ': '0C9B', 'ಜ': '0C9C', 'ಝ': '0C9D', 'ಞ': '0C9E',
    'ಟ': '0C9F', 'ಠ': '0CA0', 'ಡ': '0CA1', 'ಢ': '0CA2', 'ಣ': '0CA3',
    'ತ': '0CA4', 'ಥ': '0CA5', 'ದ': '0CA6', 'ಧ': '0CA7', 'ನ': '0CA8',
    'ಪ': '0CAA', 'ಫ': '0CAB', 'ಬ': '0CAC', 'ಭ': '0CAD', 'ಮ': '0CAE',
    'ಯ': '0CAF', 'ರ': '0CB0', 'ಱ': '0CB1', 'ಲ': '0CB2', 'ಳ': '0CB3',
    'ವ': '0CB5', 'ಶ': '0CB6', 'ಷ': '0CB7', 'ಸ': '0CB8', 'ಹ': '0CB9',
    '಼': '0CBC', 'ಽ': '0CBD',
    'ಾ': '0CBE', 'ಿ': '0CBF', 'ೀ': '0CC0', 'ು': '0CC1', 'ೂ': '0CC2',
    'ೃ': '0CC3', 'ೄ': '0CC4', 'ೆ': '0CC6', 'ೇ': '0CC7', 'ೈ': '0CC8',
    'ೊ': '0CCA', 'ೋ': '0CCB', 'ೌ': '0CCC', '್': '0CCD',
    'ೕ': '0CD5', 'ೖ': '0CD6', '೜': '0CDC', 'ೝ': '0CDD', 'ೞ': '0CDE',
    'ೠ': '0CE0', 'ೡ': '0CE1', 'ೢ': '0CE2', 'ೣ': '0CE3',
    '೦': '0CE6', '೧': '0CE7', '೨': '0CE8', '೩': '0CE9', '೪': '0CEA',
    '೫': '0CEB', '೬': '0CEC', '೭': '0CED', '೮': '0CEE', '೯': '0CEF',
    'ೱ': '0CF1', 'ೲ': '0CF2', 'ೳ': '0CF3',
}

UNICODE_TO_KANNADA = {v: k for k, v in KANNADA_UNICODE_CONSORTIUM.items()}


def get_character_unicode_info(char):
    return KANNADA_UNICODE_CONSORTIUM.get(char, None)


def validate_vocabulary_with_consortium(vocab_dict):
    """Check every vocabulary character against the Kannada Unicode table."""
    enhanced_vocab = {
        'idx_to_char': vocab_dict.get('idx_to_char', {}),
        'char_to_idx': vocab_dict.get('char_to_idx', {}),
        'vocab_size': vocab_dict.get('vocab_size', 0),
        'consortium_codes': {},
        'valid_chars': 0,
        'invalid_chars': 0,
    }
    for idx, char in vocab_dict.get('idx_to_char', {}).items():
        unicode_code = get_character_unicode_info(char)
        if unicode_code:
            enhanced_vocab['consortium_codes'][char] = unicode_code
            enhanced_vocab['valid_chars'] += 1
        else:
            enhanced_vocab['invalid_chars'] += 1
            print(f"  ⚠️  Character '{char}' (idx: {idx}) not in Unicode table")
    return enhanced_vocab


class ConsortiumLabelConverter:
    """Maps model outputs back to Kannada text. The CTC blank index is vocab_size."""

    def __init__(self, vocab_dict, verbose=True):
        self.idx_to_char = {int(k): v for k, v in vocab_dict['idx_to_char'].items()}
        self.char_to_idx = {v: int(k) for k, v in vocab_dict['idx_to_char'].items()}
        self.vocab_size = vocab_dict['vocab_size']
        self.blank_idx = self.vocab_size
        self.char_to_unicode = {}
        self.unicode_to_char = {}
        for char in self.idx_to_char.values():
            unicode_code = get_character_unicode_info(char)
            if unicode_code:
                self.char_to_unicode[char] = unicode_code
                self.unicode_to_char[unicode_code] = char
        if verbose:
            print(f"  ✅ Unicode table validated: {len(self.char_to_unicode)}/{self.vocab_size} chars")

    def get_unicode_code(self, char):
        return self.char_to_unicode.get(char, None)

    def decode(self, preds: torch.Tensor) -> list:
        """Greedy CTC decode: argmax per step, merge repeats, drop blanks."""
        argmax = preds.argmax(dim=2)
        argmax = argmax.transpose(0, 1).cpu().numpy()
        results = []
        for row in argmax:
            chars = []
            prev = -1
            for idx in row:
                if idx != prev and idx != self.blank_idx:
                    if idx in self.idx_to_char:
                        chars.append(self.idx_to_char[idx])
                prev = idx
            results.append(''.join(chars))
        return results

    def beam_search_decode(self, preds: torch.Tensor, beam_width=5) -> list:
        """Simple beam search CTC decode (expects probabilities, not log-probs)."""
        results = []
        preds_np = preds.float().cpu().numpy()  # (T, B, C)
        T, B, C = preds_np.shape

        for b in range(B):
            probs = preds_np[:, b, :]  # (T, C)
            beams = {tuple(): 1.0}

            for t in range(T):
                new_beams = defaultdict(float)
                for prefix, prob in beams.items():
                    for c in range(C):
                        if c == self.blank_idx:
                            new_beams[prefix] += prob * probs[t, c]
                        else:
                            if len(prefix) > 0 and prefix[-1] == c:
                                new_prefix = prefix
                            else:
                                new_prefix = prefix + (c,)
                            new_beams[new_prefix] += prob * probs[t, c]

                beams = dict(sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width])

            best_prefix = max(beams.items(), key=lambda x: x[1])[0]
            chars = [self.idx_to_char.get(idx, '') for idx in best_prefix if idx in self.idx_to_char]
            results.append(''.join(chars))

        return results
