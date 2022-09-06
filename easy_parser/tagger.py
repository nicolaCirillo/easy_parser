import codecs
import warnings

from spacy_conll import init_parser

class Tagger:
    def __init__(self, lang: str):
        self.nlp = init_parser(
            lang, "stanza",
            parser_opts={"use_gpu": True, "verbose": False},
            include_headers=True)

    def tag_string(self, string: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokens = self.nlp(string)
        return ' '.join(t.text + '_'+ t.pos_ for t in tokens)

    def tag_doc(self, doc: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conll = self.nlp(doc)._.conll_str
            return conll
