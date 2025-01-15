import os
import re
import sys
from threading import Lock

import phonemizer
import torch

# Add Kokoro-82M to Python path
kokoro_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Kokoro-82M')
sys.path.append(kokoro_path)
from models import build_model

# Create locks for thread-safe access
phonemizer_lock = Lock()
model_lock = Lock()

def split_num(num):
    num = num.group()
    if '.' in num:
        return num
    elif ':' in num:
        h, m = [int(n) for n in num.split(':')]
        if m == 0:
            return f"{h} o'clock"
        elif m < 10:
            return f'{h} oh {m}'
        return f'{h} {m}'
    year = int(num[:4])
    if year < 1100 or year % 1000 < 10:
        return num
    left, right = num[:2], int(num[2:4])
    s = 's' if num.endswith('s') else ''
    if 100 <= year % 1000 <= 999:
        if right == 0:
            return f'{left} hundred{s}'
        elif right < 10:
            return f'{left} oh {right}{s}'
    return f'{left} {right}{s}'

def flip_money(m):
    m = m.group()
    bill = 'dollar' if m[0] == '$' else 'pound'
    if m[-1].isalpha():
        return f'{m[1:]} {bill}s'
    elif '.' not in m:
        s = '' if m[1:] == '1' else 's'
        return f'{m[1:]} {bill}{s}'
    b, c = m[1:].split('.')
    s = '' if b == '1' else 's'
    c = int(c.ljust(2, '0'))
    coins = f"cent{'' if c == 1 else 's'}" if m[0] == '$' else ('penny' if c == 1 else 'pence')
    return f'{b} {bill}{s} and {c} {coins}'

def point_num(num):
    a, b = num.group().split('.')
    return ' point '.join([a, ' '.join(b)])

def normalize_text(text):
    text = text.replace('…', '...')  # handle unicode ellipsis
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace('«', chr(8220)).replace('»', chr(8221))
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    text = text.replace('(', '«').replace(')', '»')
    for a, b in zip('、。！，：；？', ',.!,:;?'):
        text = text.replace(a, b+' ')
    text = re.sub(r'[^\S \n]', ' ', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'(?<=\n) +(?=\n)', '', text)
    text = re.sub(r'\bD[Rr]\.(?= [A-Z])', 'Doctor', text)
    text = re.sub(r'\b(?:Mr\.|MR\.(?= [A-Z]))', 'Mister', text)
    text = re.sub(r'\b(?:Ms\.|MS\.(?= [A-Z]))', 'Miss', text)
    text = re.sub(r'\b(?:Mrs\.|MRS\.(?= [A-Z]))', 'Mrs', text)
    text = re.sub(r'\betc\.(?! [A-Z])', 'etc', text)
    text = re.sub(r'(?i)\b(y)eah?\b', r"\1e'a", text)
    text = re.sub(r'\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)', split_num, text)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = re.sub(r'(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b', flip_money, text)
    text = re.sub(r'\d*\.\d+', point_num, text)
    text = re.sub(r'(?<=\d)-(?=\d)', ' to ', text)
    text = re.sub(r'(?<=\d)S', ' S', text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", 's', text)
    text = re.sub(r'(?:[A-Za-z]\.){2,} [a-z]', lambda m: m.group().replace('.', '-'), text)
    text = re.sub(r'(?i)(?<=[A-Z])\.(?=[A-Z])', '-', text)
    return text.strip()

def get_vocab():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts

VOCAB = get_vocab()
def tokenize(ps):
    return [i for i in map(VOCAB.get, ps) if i is not None]

phonemizers = dict(
    a=phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True),
    b=phonemizer.backend.EspeakBackend(language='en-gb', preserve_punctuation=True, with_stress=True),
)

def phonemize(text, lang, norm=True, debug=False):
    """Convert text to phonemes."""
    if debug:
        print("\nKokoro.py: Debug phonemize input:")
        print(f"Text: [{text}]")
        print(f"Length: {len(text)}")
        print(f"Newlines: {text.count('\n')}")
        print(f"Special chars: {[c for c in text if not c.isalnum() and not c.isspace()]}")
    if norm:
        text = normalize_text(text)

        if debug:
            print("\nKokoro.py: After normalize_text:")
            print(f"Text: [{text}]")
            print(f"Length: {len(text)}")
            print(f"Newlines: {text.count('\n')}")
    
    # Normalize newlines and ensure text is a single line for phonemization
    text = re.sub(r'\s*\n\s*', ' ', text).strip()
    if debug:
        print("\nKokoro.py: After whitespace normalization:")
        print(f"Text: [{text}]")
        print(f"Length: {len(text)}")
        print(f"Newlines: {text.count('\n')}")
        
    try:
        if debug:
            print("\nKokoro.py: Calling phonemizer backend...")
        
        # Use the lock only around the phonemizer call
        with phonemizer_lock:
            ps = phonemizers[lang].phonemize([text])
        
        if debug:
            print(f"Raw phonemizer output: {ps}")
            print(f"Output type: {type(ps)}")
            print(f"Number of lines in output: {len(ps) if ps else 0}")
            if ps:
                for i, line in enumerate(ps):
                    print(f"Line {i}: [{line}]")
            
        if not ps:
            raise RuntimeError("Empty phonemizer output")
            
        # Check for line count mismatch
        input_lines = 1  # Since we normalized to single line
        output_lines = len(ps)
        if input_lines != output_lines:
            raise RuntimeError(f"Line count mismatch: input={input_lines}, output={output_lines}")
            
        ps = ps[0] if ps else ''
        # https://en.wiktionary.org/wiki/kokoro#English
        ps = ps.replace('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ').replace('kəkˈɔːɹəʊ', 'kˈəʊkəɹəʊ')
        ps = ps.replace('ʲ', 'j').replace('r', 'ɹ').replace('x', 'k').replace('ɬ', 'l')
        ps = re.sub(r'(?<=[a-zɹː])(?=hˈʌndɹɪd)', ' ', ps)
        ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»"" ]|$)', 'z', ps)
        if lang == 'a':
            ps = re.sub(r'(?<=nˈaɪn)ti(?!ː)', 'di', ps)
        ps = ''.join(filter(lambda p: p in VOCAB, ps))
        
        if debug:
            print("\nKokoro.py: Final phonemes:")
            print(f"Phonemes: [{ps}]")
            print(f"Length: {len(ps)}")
            print(f"Output lines: {ps.count('\n') + 1}")
            
        return ps.strip()
        
    except Exception as e:
        if debug:
            print(f"\nKokoro.py: Error in phonemize: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Text that caused error: [{text}]")
            print(f"Text length: {len(text)}")
            print(f"Text newlines: {text.count('\n')}")
            print(f"Text special chars: {[c for c in text if not c.isalnum() and not c.isspace()]}")
        raise

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

@torch.no_grad()
def forward(model, tokens, ref_s, speed):
    device = ref_s.device
    tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long()
    pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item())
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + pred_dur[0,i].item()] = 1
        c_frame += pred_dur[0,i].item()
    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(tokens, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
    return model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu().numpy()

def generate(model, text, voicepack, lang='a', speed=1, ps=None):
    ps = ps or phonemize(text, lang)
    tokens = tokenize(ps)
    if not tokens:
        return None
    elif len(tokens) > 510:
        print(f'Segment too long: {len(tokens)} tokens')
        return None
    ref_s = voicepack[len(tokens)]
    out = forward(model, tokens, ref_s, speed)
    ps = ''.join(next(k for k, v in VOCAB.items() if i == v) for i in tokens)
    return out, ps
