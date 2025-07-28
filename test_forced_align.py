import argparse
import re
import torch
import torchaudio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
)
from transformers.models.wav2vec2_phoneme.tokenization_wav2vec2_phoneme import (
    Wav2Vec2PhonemeCTCTokenizer,
)


def run_alignment(wav_path: str, transcript: str):
    """Run forced alignment on a single audio file and transcript."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    )
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
        "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        espeak_path="espeak",
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    )
    model.eval()

    waveform, sr = torchaudio.load(wav_path)
    target_sr = feature_extractor.sampling_rate
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    with torch.no_grad():
        inputs = feature_extractor(
            waveform.squeeze(), sampling_rate=target_sr, return_tensors="pt"
        )
        logits = model(inputs.input_values).logits
        emission = torch.log_softmax(logits, dim=-1)

    # Normalize transcript by removing punctuation and dropping unknown tokens
    def normalize(text: str):
        cleaned = []
        for tok in text.split():
            tok = re.sub(r'[!.,?"«»]', '', tok)
            if not tok:
                continue
            if tok in tokenizer.get_vocab():
                cleaned.append(tok)
            else:
                print(f"Warning: dropping unsupported token '{tok}'")
        return ' '.join(cleaned)

    transcript = normalize(transcript)
    token_ids = tokenizer(transcript, add_special_tokens=False).input_ids
    if emission.size(1) < len(token_ids):
        raise ValueError(
            f"Transcript is too long: {len(token_ids)} tokens for {emission.size(1)} frames."
        )
    tokens = torch.tensor([token_ids], dtype=torch.int32)
    alignments, scores = torchaudio.functional.forced_align(
        emission, tokens, blank=tokenizer.pad_token_id
    )
    alignments, scores = alignments[0], scores[0].exp()

    # Print token for each frame with probability score
    labels = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))
    for idx, (a, s) in enumerate(zip(alignments.tolist(), scores.tolist())):
        label = labels[a]
        print(f"frame {idx:04d}: {label}\tscore={s:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forced alignment test")
    parser.add_argument("wav", help="Path to WAV file")
    parser.add_argument(
        "transcript",
        help="Space-separated phonemes produced by eSpeak for the audio",
    )
    args = parser.parse_args()
    run_alignment(args.wav, args.transcript)
