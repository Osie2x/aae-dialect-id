import json,re,numpy as np,pandas as pd,sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
U=sys.argv[1]
rows=[json.loads(l) for l in open(U+"/ab9faa8e-aae_example_bank.jsonl")]
pairs={}
for r in rows:
    a,s=(r.get("aae_original") or "").strip(),(r.get("sae_translation") or "").strip()
    if not a or not s or r.get("original_is_natural_aae")!="natural": continue
    if re.sub(r"\W","",a.lower())==re.sub(r"\W","",s.lower()): continue
    pairs[a]=s
P=list(pairs.items()); print("twitter pairs",len(P))
X=[a for a,s in P]+[s for a,s in P]; y=[1]*len(P)+[0]*len(P); g=list(range(len(P)))*2
def model(): return make_pipeline(make_union(TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=2,sublinear_tf=True),
    TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True)),LogisticRegression(C=4,max_iter=2000))
X,y,g=np.array(X,dtype=object),np.array(y),np.array(g)
accs=[]
for tr,te in GroupShuffleSplit(5,test_size=.2,random_state=0).split(X,y,g):
    m=model().fit(X[tr],y[tr]); accs.append(accuracy_score(y[te],m.predict(X[te])))
print("random split (Twitter AAE vs its SAE), 5 seeds: %.3f +- %.3f"%(np.mean(accs),np.std(accs)))
m=model().fit(X,y)
d=pd.read_csv(U+"/62540e2d-AAE_Research_Dataset_Expanded_2026_09_30.csv",low_memory=False)
pos=d[(d.dialect_label=="AAE")&(d.source_subtype!="twitteraae_blodgett")]
SLANG=r"\b(ain'?t|aint|finna|ion|lowkey|imma|ima|i'?ma|gon|gonna|tryna|bout|nah|yall|y'all|lol|lmao|smh|fr|bruh|shit|nigga\w*)\b"
mask=lambda t: re.sub(SLANG,"",t,flags=re.I)
out=[]
for src,grp in pos.groupby("source_subtype"):
    t=grp.item_text.astype(str).tolist()
    out.append((src,len(t),m.predict(t).mean(),m.predict([mask(x) for x in t]).mean()))
tw=X[y==1]; out.append(("twitter (train, in-domain)",len(tw),m.predict(tw).mean(),m.predict([mask(x) for x in tw]).mean()))
print(pd.DataFrame(out,columns=["source","n","recall","recall_slang_masked"]).round(3).to_string(index=False))
# negatives: SAE side masked false positive rate
sae=X[y==0]; print("SAE flagged as AAE (train): %.3f"%m.predict(sae).mean())
# top features
vec=m[0]; names=np.concatenate([t.get_feature_names_out() for _,t in vec.transformer_list]); coef=m[1].coef_[0]
print("top AAE features:",[names[i] for i in np.argsort(-coef)[:25]])
