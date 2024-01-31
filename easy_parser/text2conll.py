from tqdm import tqdm
import pandas as pd
import codecs
import os


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


def lines2conll(txtfile: str, fileout: str, lang: str, encoding="utf8", 
                verbose=False):
     tagger = Tagger(lang)
     with codecs.open(txtfile, "r", encoding) as filein:
          with codecs.open(fileout, "w", "utf8") as writer:
            if verbose: filein = tqdm(filein, desc="documents parsed:")
            for i, line in enumerate(filein):
                head = f"# newdoc id = {i}\n"
                conll = tagger.tag_doc(line.strip()) + "\n"
                writer.write(head  + conll)

def folder2conll(folder: str, fileout: str, lang: str, encoding="utf8", 
                verbose=False):
      tagger = Tagger(lang)
      txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
      if verbose: 
          txt_files = tqdm(txt_files, desc="documents parsed:")
      with codecs.open(fileout, "w", "utf8") as writer:
           for f in txt_files:
            with codecs.open(f"{folder}/{f}", "r", encoding) as filein:
                doc = filein.read()
                head = f"# newdoc id = {f.rsplit('.', 1)}\n"
                conll = tagger.tag_doc(doc.strip()) + "\n"
                writer.write(head + conll)