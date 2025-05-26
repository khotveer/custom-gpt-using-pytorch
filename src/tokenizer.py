import os
import json
import regex as re
import requests

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:] 
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    return d

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def get_file(local_file, remote_file):
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        open(local_file, "wb").write(response.content)


class Encoder:

    def __init__(self, encoder, bpe_merges):
        # byte encoder/decoder
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        
        # bpe token encoder/decoder
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        # bpe merge list that defines the bpe "tree", of tuples (a,b) that are to merge to token ab
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bpe(self, token):
        # token is a string of one individual 'word', after byte encoding, e.g. 'Ġthere'

        # memoization, for efficiency
        if token in self.cache:
            return self.cache[token]

        word = tuple(token) # individual characters that make up the token, in a tuple
        pairs = get_pairs(word) # get all bigrams

        if not pairs:
            return token

        while True:

            # find the next lowest rank bigram that can be merged
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break # no more bigrams are eligible to be merged
            first, second = bigram

            # we will now replace all occurences of (first, second) in the list of current
            # words into one merged token first_second, in the output list new_words
            new_word = []
            i = 0
            while i < len(word):

                # find the next occurence of first in the sequence of current words
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                # if this occurence is also followed by second, then merge them into one
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # all occurences of (first, second) have been merged to first_second
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # concat all words into a string, and use ' ' as the separator. Note that
        # by now all characters have been byte encoded, guaranteeing that ' ' is
        # not used in the actual data and is a 'special' delimiter character
        word = ' '.join(word)

        # cache the result and return
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_idx = []
        # pre-tokenize the input text into string tokens (words, roughly speaking)
        tokens = re.findall(self.pat, text)
        # process each token into BPE integers
        for token in tokens:
            # encode the token as a bytes (b'') object
            token_bytes = token.encode('utf-8')
            # translate all bytes to their unicode string representation and flatten
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            # perform all the applicable bpe merges according to self.bpe_ranks
            token_merged = self.bpe(token_translated).split(' ')
            # translate all bpe tokens to integers
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            # extend our running list of all output integers
            bpe_idx.extend(token_ix)
        return bpe_idx

    def encode_and_show_work(self, text):
        bpe_idx = []
        parts = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            parts.append({
                'token': token,
                'token_bytes': token_bytes,
                'token_translated': token_translated,
                'token_merged': token_merged,
                'token_ix': token_ix,
            })
        out = {
            'bpe_idx': bpe_idx, # the actual output sequence
            'tokens': tokens, # result of pre-tokenization
            'parts': parts, # intermediates for each token part
        }
        return out

    def decode(self, bpe_idx):
        # inverse map the integers to get the tokens
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        # inverse the byte encoder, e.g. recovering 'Ġ' -> ' ', and get the bytes
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        # recover the full utf-8 string
        text = tokens_bytes.decode('utf-8', errors='replace')
        return text

class BPE:
    def __init__(self):
        self.encoder, self.decoder, self.vocab_size = self.get_encoder()
        # self.decoder = enc_obj
    
    def get_file(self, local_file, remote_file):
        if not os.path.isfile(local_file):
            print(f"downloading {remote_file} to {local_file}")
            response = requests.get(remote_file)
            open(local_file, "wb").write(response.content)

    def get_the_encoder(self):

        directory = "./temp/"
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # downloading the pairs, which is used in GPT-2 model
        encoder_local_file = os.path.join(directory, 'encoder.json')
        encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
        self.get_file(encoder_local_file, encoder_remote_file)
        
        with open("./temp/encoder.json", 'r') as f:
            encoder = json.load(f)
        
        assert (len(encoder) == 50257), "Encoder length donwloaded is not matching 50257"
        
        # donwloading the vocab.bpe rules
        vocab_local_file = os.path.join(directory, 'vocab.bpe')
        vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
        self.get_file(vocab_local_file, vocab_remote_file)
        with open(vocab_local_file, 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        
        assert (len(bpe_merges) == 50000), "BPE length donwloaded is not matching 50000"
        
        encoder = Encoder(encoder, bpe_merges)

        return encoder

    def get_encoder(self):
        enc_dec_obj = self.get_the_encoder()
        encoder = enc_dec_obj.encode
        decoder = enc_dec_obj.decode
        return encoder, decoder, len(enc_dec_obj.encoder)

