from tqdm import tqdm
import pandas as pd
import codecs


from .tagger import Tagger

DEFAULT = {'sep': ';'}

def csv2conll(csvfile: str, fileout: str, lang: str, doc_col: str,  
              id_col: str = None, verbose=False, csvparams=DEFAULT):
            
            df = pd.read_csv(csvfile, **csvparams)
            docs = df[doc_col]
            if id_col:
                ids = df[id_col]
            else:
                ids = ['doc_{}'.format(i+1) for i in range(0, len(docs))]
            tagger = Tagger(lang)
            with codecs.open(fileout, 'w', 'utf8') as writer:
                id_doc = list(zip(ids, docs))
                if verbose: id_doc = tqdm(id_doc, desc="documents parsed:")
                for id, doc in id_doc:
                    head = "# newdoc id = {}\n".format(id)
                    sents = doc.split('\n')
                    conll = ''
                    for s in sents:
                        conll += tagger.tag_doc(s) + '\n'
                    writer.write(head + conll)


            

