import pandas as pd
from sentence_transformers import SentenceTransformer as ST
import sklearn.cluster as CLU
import sklearn.decomposition as DEC
import sklearn.preprocessing as PRE
from sklearn.feature_extraction.text import CountVectorizer
import spacy

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, cdist
import numpy as np
import os

from numpy import dot
from numpy.linalg import norm

def CS(a,b):
    
    return dot(a,b)/(norm(a)*norm(b))

class TextCluster:
    
    def __init__(self, ClusterTexts:list, OutDimensions=None,clustering='OPTICS',
                 metric='cosine', min_samp_frac=0.002, samp_min=5, max_eps=1.5,
                 eps=0.8, xi=0.0001, WriteEncodings=False, WriteLabels=False,
                 ReadEncodings=False, ReadLabels=False, IntermediatePath='Intermediates.json',
                 LabelPath='Clusters.json', MinSimilThresh=0.5, MaxSeparateSimilarity=0.7,
                 MergeNearbyClusters=True):
        
        # Simply a list of texts to cluster. Could be sentences or paragraphs,
        # although the maximum intake for sentence embedders is 512 tokens.
        # The incoming texts recieve no further preprocessing, anything coming
        # should have preprocessing steps done prior.
        self.ClusterTexts = ClusterTexts
        
        # OutDimensions control the dimensions of the embedded output. Dimensionality
        # reduction is accomplished using Sci-Kit Learn's implementation of PCA
        # If None or greater than the incoming dimension, the embeddings are left alone.
        self.OutDimensions = OutDimensions
        
        # These variable control the implementation of the clustering algorithm
        # By default, OPTICS is used. Optics tends to result in small clusters,
        # but also tends to avoid the problem of clusters quickly clumping together.
        self.clustering = clustering
        self.metric = metric
        self.min_samp_frac = min_samp_frac
        self.samp_min = samp_min
        self.max_eps = max_eps
        self.xi = xi
        self.eps = eps
        
        # Depending on the size of your cluster list, encoding and clustering
        # can both be time consuming. Technically, if you are doing the encoding,
        # you should also be doing the clustering because any changes in encoding
        # should change your clusters. The converse is not true as you can test different
        # cluster settings
        self.ReadEncodings = ReadEncodings
        self.ReadLabels = ReadLabels
        self.WriteEncodings = WriteEncodings
        self.WriteLabels = WriteLabels
        self.IntermediatePath = IntermediatePath
        self.LabelPath = LabelPath
        
        # This controls the cluster merging aspect, which is especially helpful
        # when using OPTICS as the clustering algorithm
        self.MergeNearbyClusters = MergeNearbyClusters
        self.MinSimilThresh = MinSimilThresh
        self.MaxSeparateSimilarity = MaxSeparateSimilarity
        
        
        self.data = pd.DataFrame({'Texts':self.ClusterTexts})
        self.data['Encodings'] = None
        self.data['Label'] = None
        
        self.RVdata = None
        
    def PrepAndEncode(self):
        
        # Checks if we want to read encodings and if the stated file path exists
        # If not both, continue with encoding
        if os.path.isfile(self.IntermediatePath) and self.ReadEncodings:
            
            self.data = pd.read_json(self.IntermediatePath)
            return
        
        # Faster but still high performance embedder. I could potentially give
        # a choice here, but I don't necessarily see the need. The convert_to_numpy
        # is probably not strictly necessary, but it also isn't costly.  
        model = ST('all-MiniLM-L6-v2')
        temp = model.encode(self.ClusterTexts, convert_to_numpy=True, normalize_embeddings=True)
        
        if self.OutDimensions:
            if self.OutDimensions<temp.shape[1]:
                
                # Using a fully deterministic svd_solver helps with reproducibility.
                # The typical algorithm used for any matrix you would see in with
                # text embeddings would be random. This results in some noise around
                # what exactly falls into a cluster and if some smaller clusters form.
                # I generally find the behavior undesirable and so sacrifice some speed
                # to get reproducible outputs.
                pca = DEC.PCA(n_components=self.OutDimensions, svd_solver='arpack')
                pca.fit_transform(temp)
        
        Encodings = temp

        # This is a grossly inefficient abuse of Pandas. That said, it has the benefit
        # of circumventing any behavior related to how Pandas struggles to write lists
        # of lists (or equivalent arrays) to columns, and it is so relatively fast compared
        # to the embedding process that it isn't important.
        for i in self.data.index:
            self.data.at[i,'Encodings'] = Encodings[i]
        
        if self.WriteEncodings:
            self.data.to_json(self.IntermediatePath)
    
        
        
    def Cluster(self):
        
        # Again, checking if we want to read and if the path is a file.        
        if os.path.isfile(self.LabelPath) and self.ReadLabels:
            
            self.data = pd.read_json(self.LabelPath)
            return
        
        # Selects the Minimum Sample size. There is a bit of a twist because
        # your minimum samples should probably be higher if you use DBSCAN
        # but that is for the users to know or learn.
        ms = max(len(self.data)*self.min_samp_frac,self.min_samp)

        # Could potentially add more options here. Also, yes, my choice of 
        if self.clustering.lower() == 'optics':
            DB = CLU.OPTICS(min_samples = int(ms), xi=self.xi, max_eps=self.max_eps, metric=self.metric)
        else:
            DB = CLU.DBSCAN(eps=self.eps, min_samples=int(ms), metric=self.metric)
            
        DB.fit(self.data['Encodings'].tolist())
        
        labels = DB.labels_
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)        

        self.data['Label'] = DB.labels_
        
        if self.MergeNearbyClusters:
            # Runs until it hits a breaking point
            while True:
                RVs = []
                
                # Gets the unique labels that aren't noise
                alllabs = pd.unique(self.data['Label'])
                alllabs = [i for i in alllabs if not i==-1]
                print(alllabs)
                
                # Calculates all the Representative Vectors for each cluster
                for label in alllabs:
                    
                    temp = self.WebContentData[self.data['Label']==label]
                    RVs.append(np.mean(temp['Encodings'].tolist(), axis=0))
                
                # Calculates all pairwise similarity between all representative
                # vectors. Excludes self-to-self pairings by setting the similarity
                # to zero
                pairwise = 1-cdist(RVs,RVs,'cosine')
                pairwise[pairwise>0.999] = 0
                
                # This merges the culsters of the most similar representative
                # vectors together. Strictly speaking, due to the nature of how
                # OPTICS clustering works, this doesn't mean that you end up
                # with one big cluster. In fact, the representative vector for
                # the merged cluster is likely to be in a lower density region
                # than the representative vectors for the two clusters. However,
                # the expansion technique in the next method render most of this
                # moot, as it includes these low density vector near the clusters
                # anyways.
                top = np.max(pairwise)
                if top>self.MaxSeparateSimilarity:
                    ProblemIndicies = np.argwhere(pairwise==top)
                
                    self.data.loc[self.data['Label']==alllabs[ProblemIndicies[0][1]],'Label'] = alllabs[ProblemIndicies[0][0]]
                else:
                    break
        else:
            alllabs = pd.unique(self.data['Label'])
            alllabs = [i for i in alllabs if not i==-1]
            RVs = []
            for label in alllabs:
                
                temp = self.WebContentData[self.data['Label']==label]
                RVs.append(np.mean(temp['Encodings'].tolist(), axis=0))
            
        self.RVData = pd.DataFrame({'label':alllabs})
        self.RVData['RepresentativeVector'] = None
        
        RVs = PRE.normalize(RVs).tolist()
        
        for i in self.WebContentRVData.index:
            self.RVData.at[i,'RepresentativeVector'] = RVs[i]

        if self.WriteLabels:
            self.data.to_json(self.LabelPath)
        
    def ExpandSents(self):
        
        self.RVData['Sentences'] = None
        self.RVData['Similarities'] = None
        
        # Why do I use sklearn's pairwise_distance here but scipy's cdist other
        # places. That is a good question.
        Dists = pairwise_distances(np.array(self.RVdata['RepresentativeVector'].tolist()), metric='cosine')
        
        # This helps to set boundaries for clusters. having self.DistFrac<0.5 
        # guarantees that clusters will be mutually exclusive. However, some
        # applications may be okay with clusters that overlap, and the merge.
        # Additionally, vectors that are closer to each other have smaller
        # "gather" distances so that the overlap isn't extreme even when
        # self.DistFrac=1
        Dists = 1-self.DistFrac*np.min(Dists, axis=0, where=Dists>0, initial=2)
        Dists = [i if i>self.MinSimilThresh else self.MinSimilThresh for i in Dists]
        
        self.RVdata['MinimumSimilarity'] = Dists
        
        for i in self.RVdata.index:
            
            MinSim = self.RVdata.at[i,'MinimumSimilarity']
            Vec = self.RVdata.at[i,'RepresentativeVector']
            
            Embeddings = self.data['Encodings'].tolist()
            
            # Calculates the similarity of each individual text embedding. This
            # not only allows us to filter what should be associated with a particular
            # label, but also a way to rank those by the most related. Now, as mentioned
            # above, the center of a "cluster" may actually be somewhat sparse if
            # clusters were merged, but we'll consider it good enough.
            self.data['Similarity'] = [CS(Vec, Embed) for Embed in Embeddings]
            
            tempDF = self.data[self.data['Similarity']>MinSim]
            tempDF = tempDF.sort_values('Similarity', ascending=False)
            
            self.RVdata.at[i,'Sentences'] = tempDF['Response'].tolist()
            self.RVdata.at[i,'Similarities'] = tempDF['Similarity'].tolist()
            
        return self.RVdata
        
    def FindTopicWords(self):

        # This function adds the top ten topic words, using an algorithm that is
        # absolutely horrendous and could likely be replaced by a much simpler
        # TF-IDF algorithm.
        
        SPACYpiplin = spacy.load("en_core_web_lg", disable=['parser', 'ner'])
        
        TopicWordList = []
        
        WholeDoc = SPACYpiplin.pipe(self.data['EmbeddingInputs'].tolist())
        WholeRec = ' '.join([' '.join([token.lemma_ for token in doc if not token.is_stop]) for doc in WholeDoc])

        CV = CountVectorizer(decode_error='ignore')
        
        X0 = CV.fit_transform([WholeRec]).toarray()
        
        vocab = CV.vocabulary_
        
        for label in self.RVdata['label']:
                   
            LabelDoc = SPACYpiplin.pipe(self.data.loc[self.data['labels']==label,'EmbeddingInputs'].tolist())
            
            LabelRec = ' '.join([' '.join([token.lemma_ for token in doc if not token.is_stop]) for doc in LabelDoc])
            
            CV = CountVectorizer(decode_error='ignore', vocabulary=vocab)

            
            X = CV.fit_transform([LabelRec]).toarray()
            Ratios =  np.divide(X[0],X0[0])
            
            InverseTranslator = {v: k for k, v in CV.vocabulary_.items()}
            
            tokenlist = [InverseTranslator[i] for i in range(len(Ratios))]
            Fraclist = [Ratios[i] for i in range(len(Ratios))]
            Occurences = [X0[0][i] for i in range(len(Ratios))]
            
            TempDF = pd.DataFrame({'Token': tokenlist, 'Ratio': Fraclist,
                                   'TotalOccurences':Occurences})
            
            TempDF = TempDF[TempDF['TotalOccurences']>len(self.data)/1000]
            
            TempDF = TempDF.sort_values('Ratio',ascending=False)
            
            TempDF = TempDF.head(10).sort_values('TotalOccurences', ascending=False)
            
            TopicWords = TempDF['Token'].tolist()
            TopicWordList.append(TopicWords)
            
            
        self.RVdata.insert(1, 'TopicWords', TopicWordList)
        
        return self.RVdata


