# concept_embeddings
This is an experimental project to model linguistic concepts based on word embeddings.

## Approach

Try to represent semantic concepts as vectors. Infer this vectors from word embeddings.

A concept may correspond to one word or to a sequence of words. Examples:
  - *star*
  - *solar system*

Inversely, one word or sequence of words may represent one or more concepts.
In other words, there exists a *n - n* relation between sequences of words and concepts.

Represent one concept as a vector, composed of two parts:  
  - An *internal component* (`C_i`): 
    - The embedding corresponding to the word or the sequence of words representing the concept.
    - If more than one word, use pooling.
  - An *external component* (`C_o`):
    - The embedding corresponding to the typical context where the concept is usually located.
      One option is to represent this context using a pooling of the embedding vectors corresponding
      to the context words. 
Therefore, `C_raw = [C_i, C_o]`, where `C_raw` is the raw concept vector.

**NOTE**: If a contextual embedding is used the `C_o` component may note be necessary, since these embeddings already
take into account the wor context.

Cluster the raw concept vectors for each word or sequence of words. Hopefully, each cluster 
each vector will correspond to the one different semantic concept.

Finally, concepts will be modelled with one representative vector for each cluster, such 
as the cluster centroid. In fact, this process will try to quantify concepts, assigning them 
a single (embedding) vector. Hopefully, the distance between the vectors of two concepts will 
give us an estimation of their semantic distance, keeping a **discrete concept space**.


## Steps

1. Use word embeddings as the basis of representation:
    1. Use word embeddings
    2. Use contextual embeddings
2. Identify candidate words or word sequences to analyse.
3. Calculate raw vectors for each candidate: `C_raw = [C_i, C_o]`.
4. Cluster the raw vectors of candidates.
5. Calculate the representatives of the identified clusters for each candidate.
6. Identify synonym relations: search for equivalent concepts: most similar representatives.
7. Assign symbols (identifiers) to every cluster: concepts.

Finally, a concept will be represented by a pair
`(Concept_id, C)`, where `C` is the representative vector of the cluster of `C_raw` vectors 
an `Concept_id` is a unique identifier of the vector.

### 1. Use word embeddings as the basis of representation

Use precalculated embeddings:

1. Word embeddings:
    - FastText
2. Contextual embeddings:
    - Bert
    - Bert-like

### 2. Identify candidate words or word sequences to analyse

One possible approach: select an ontology and model its concepts.

**TODO**

### 3. Calculate raw vectors for each candidate

Look for phrases / paragraphs where each word sequence appear.
One possible approach: use Solr to look for text fragments.

**TODO**

### 4. Cluster the raw vectors of candidates

**TODO**

### 5. Calculate the representatives of the identified clusters for each candidate

**TODO**

### 6. Identify synonym relations

**TODO**

### 7. Assign symbols (identifiers) to every cluster

