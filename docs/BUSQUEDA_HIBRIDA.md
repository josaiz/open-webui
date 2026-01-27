# Búsqueda Híbrida en Open WebUI

## Índice

1. [Implementación Técnica Actual](#1-implementación-técnica-actual)
2. [Guía de Onboarding: Entendiendo la Búsqueda Híbrida](#2-guía-de-onboarding-entendiendo-la-búsqueda-híbrida)
3. [Comparativa: Búsqueda Híbrida In-Process vs Microservicio Dedicado](#3-comparativa-búsqueda-híbrida-in-process-vs-microservicio-dedicado)

---

## 1. Implementación Técnica Actual

### 1.1 Arquitectura General

La búsqueda híbrida en Open WebUI combina dos estrategias complementarias de recuperación de información:

1. **BM25 (Best Matching 25)**: Búsqueda léxica/keyword-based
2. **Vector Search**: Búsqueda semántica basada en embeddings

Estas se fusionan mediante un **EnsembleRetriever** con pesos configurables, seguido de un proceso de **reranking** para optimizar los resultados finales.

### 1.2 Pipeline Completo de Búsqueda Híbrida

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE BÚSQUEDA HÍBRIDA                          │
└─────────────────────────────────────────────────────────────────────────┘

ENTRADA: Query del usuario + Colección(es) a consultar
   │
   ├──────────────────────────────────────────────────────────────────────┐
   │                                                                       │
   v                                                                       v
┌──────────────────────┐                                    ┌─────────────────────┐
│  BM25 RETRIEVER      │                                    │ VECTOR RETRIEVER    │
│  (Búsqueda Léxica)   │                                    │ (Búsqueda Semántica)│
├──────────────────────┤                                    ├─────────────────────┤
│ • Tokenización       │                                    │ • Embedding del     │
│ • TF-IDF scoring     │                                    │   query             │
│ • Ranking por        │                                    │ • Similarity search │
│   relevancia keyword │                                    │   (cosine/euclidean)│
│ • TOP_K resultados   │                                    │ • TOP_K resultados  │
└──────────┬───────────┘                                    └──────────┬──────────┘
           │                                                           │
           │              ┌─────────────────────┐                      │
           └─────────────>│ ENSEMBLE RETRIEVER  │<─────────────────────┘
                          ├─────────────────────┤
                          │ Fusión con pesos:   │
                          │ • w_bm25 (default:  │
                          │   0.5)              │
                          │ • w_vector (default:│
                          │   0.5)              │
                          │ • Normalización RRF │
                          │   (Reciprocal Rank  │
                          │   Fusion)           │
                          └──────────┬──────────┘
                                     │
                                     v
                          ┌─────────────────────┐
                          │  RERANK COMPRESSOR  │
                          ├─────────────────────┤
                          │ OPCIÓN A:           │
                          │ • Cross-Encoder     │
                          │   (sentence-        │
                          │   transformers)     │
                          │                     │
                          │ OPCIÓN B:           │
                          │ • External Reranker │
                          │   (ColBERT, API)    │
                          │                     │
                          │ • Score threshold   │
                          │   filtering         │
                          │ • TOP_K_RERANKER    │
                          │   resultados        │
                          └──────────┬──────────┘
                                     │
                                     v
                          ┌─────────────────────┐
                          │ RESULTADOS FINALES  │
                          │ • Ordenados por     │
                          │   score             │
                          │ • Metadata incluida │
                          │ • Distances/scores  │
                          └─────────────────────┘
```

### 1.3 Componentes Principales

#### 1.3.1 BM25Retriever

**Ubicación**: `backend/open_webui/retrieval/utils.py` (líneas 249-253)

**Implementación**:
```python
bm25_retriever = BM25Retriever.from_texts(
    texts=bm25_texts,
    metadatas=collection_result.metadatas[0],
)
bm25_retriever.k = k
```

**Características**:
- Utiliza `BM25Retriever` de `langchain_community.retrievers`
- **BM25 (Okapi BM25)**: Algoritmo probabilístico de ranking basado en:
  - **TF (Term Frequency)**: Frecuencia del término en el documento
  - **IDF (Inverse Document Frequency)**: Rareza del término en la colección
  - **Normalización por longitud del documento**
- Parámetros ajustables:
  - `k1`: Controla saturación de TF (típicamente 1.2-2.0)
  - `b`: Normalización por longitud del documento (0-1)

**Textos Enriquecidos** (opcional):
- Si `ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS=True`:
  - Añade metadata al texto: `[source] snippet`
  - Mejora matching en contextos específicos

#### 1.3.2 VectorSearchRetriever

**Ubicación**: `backend/open_webui/retrieval/utils.py` (líneas 91-122)

**Implementación**:
```python
class VectorSearchRetriever(BaseRetriever):
    collection_name: Any
    embedding_function: Any
    top_k: int

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        result = await VECTOR_DB_CLIENT.search(
            collection_name=self.collection_name,
            vectors=[await self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)],
            limit=self.top_k,
        )
        # ... procesamiento de resultados
```

**Características**:
- Búsqueda semántica por **similitud vectorial**
- Embedding del query con el mismo modelo usado en indexación
- Métrica de distancia configurable por backend:
  - **Cosine similarity** (ChromaDB, Qdrant, Milvus)
  - **Euclidean distance** (pgvector, OpenSearch)
  - **Dot product** (Pinecone, Weaviate)
- Soporte para 12+ bases de datos vectoriales

#### 1.3.3 EnsembleRetriever

**Ubicación**: `backend/open_webui/retrieval/utils.py` (líneas 261-273)

**Implementación**:
```python
if hybrid_bm25_weight <= 0:
    # Solo búsqueda vectorial
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_search_retriever], 
        weights=[1.0]
    )
elif hybrid_bm25_weight >= 1:
    # Solo BM25
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever], 
        weights=[1.0]
    )
else:
    # Búsqueda híbrida real
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_search_retriever],
        weights=[hybrid_bm25_weight, 1.0 - hybrid_bm25_weight],
    )
```

**Fusión de Resultados (RRF - Reciprocal Rank Fusion)**:
```
Para cada documento d:
    score(d) = w_bm25 * (1 / (rank_bm25(d) + k)) + 
               w_vector * (1 / (rank_vector(d) + k))
    
Donde:
    - k = 60 (constante de LangChain)
    - w_bm25 + w_vector = 1.0
    - rank(d) = posición del documento en el ranking (1, 2, 3, ...)
```

**Ventajas de RRF**:
- Normalización independiente de scores absolutos
- Manejo robusto de diferentes escalas de scoring
- No requiere calibración entre BM25 y vector scores

#### 1.3.4 RerankCompressor

**Ubicación**: `backend/open_webui/retrieval/utils.py` (líneas 1259-1337)

**Implementación**:
```python
class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float  # Relevance threshold

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        # Opción 1: Reranking con función externa o cross-encoder
        if self.reranking_function is not None:
            scores = await asyncio.to_thread(
                self.reranking_function, query, documents
            )
        # Opción 2: Cosine similarity con embeddings
        else:
            query_embedding = await self.embedding_function(
                query, RAG_EMBEDDING_QUERY_PREFIX
            )
            document_embedding = await self.embedding_function(
                [doc.page_content for doc in documents], 
                RAG_EMBEDDING_CONTENT_PREFIX
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        # Filtrado por threshold
        docs_with_scores = [(d, s) for d, s in zip(documents, scores) 
                            if s >= self.r_score]
        
        # Ordenar y tomar TOP_K_RERANKER
        result = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        return result[:self.top_n]
```

**Modos de Reranking**:

1. **Cross-Encoder** (por defecto):
   - Modelos como `jinaai/jina-reranker-v2-base-multilingual`
   - Entrada: `[query, document]` → score directo
   - Más preciso que bi-encoders pero más lento
   - Usado para refinar TOP_K_RERANKER (típicamente 3-10 docs)

2. **External Reranker** (`RAG_RERANKING_ENGINE="external"`):
   - ColBERT: `jinaai/jina-colbert-v2`
   - APIs de reranking personalizadas
   - Flexibilidad para usar servicios especializados

3. **Cosine Similarity** (fallback):
   - Si no hay reranking_function
   - Re-embeddea query y documentos
   - Calcula similitud coseno

### 1.4 Configuraciones y Parámetros

#### Variables de Entorno Principales

| Variable | Tipo | Default | Descripción |
|----------|------|---------|-------------|
| `ENABLE_RAG_HYBRID_SEARCH` | bool | `true` | Activa/desactiva búsqueda híbrida |
| `RAG_HYBRID_BM25_WEIGHT` | float | `0.5` | Peso de BM25 (0.0-1.0). Vector = 1.0 - BM25 |
| `ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS` | bool | `false` | Añade metadata a textos BM25 |
| `TOP_K` | int | `10` | Documentos iniciales por retriever |
| `TOP_K_RERANKER` | int | `3` | Documentos finales tras reranking |
| `RELEVANCE_THRESHOLD` | float | `0.0` | Score mínimo para incluir documento |
| `RAG_RERANKING_MODEL` | str | `""` | Modelo de reranking (ej: `jinaai/jina-reranker-v2`) |
| `RAG_RERANKING_ENGINE` | str | `""` | `"external"` para reranker externo |

#### Configuración por Colección (Override API)

**Endpoint**: `POST /api/retrieval/query/collection/{collection_name}`

Permite override de parámetros por consulta:
```json
{
  "query": "¿Cómo funciona X?",
  "k": 20,                    // Override TOP_K
  "r": 0.3,                   // Override RELEVANCE_THRESHOLD
  "hybrid": true,             // Override ENABLE_RAG_HYBRID_SEARCH
  "hybrid_bm25_weight": 0.7   // Override RAG_HYBRID_BM25_WEIGHT
}
```

### 1.5 Flujo de Ejecución Detallado

**Función Principal**: `query_doc_with_hybrid_search()`

```python
async def query_doc_with_hybrid_search(
    collection_name: str,
    collection_result: GetResult,  # Documentos de la colección
    query: str,
    embedding_function,
    k: int,                        # TOP_K inicial
    reranking_function,
    k_reranker: int,               # TOP_K tras reranking
    r: float,                      # Relevance threshold
    hybrid_bm25_weight: float,     # Peso BM25
    enable_enriched_texts: bool = False,
) -> dict:
```

**Pasos de Ejecución**:

1. **Preparación de Textos BM25**:
   ```python
   bm25_texts = (
       get_enriched_texts(collection_result)  # Con metadata
       if enable_enriched_texts
       else collection_result.documents[0]    # Solo contenido
   )
   ```

2. **Creación de Retrievers**:
   - BM25Retriever con `k` documentos
   - VectorSearchRetriever con `k` documentos

3. **Ensamblado según Peso**:
   - `weight <= 0`: Solo vector
   - `weight >= 1`: Solo BM25
   - `0 < weight < 1`: Híbrido con RRF

4. **Compresión con Reranking**:
   ```python
   compressor = RerankCompressor(
       embedding_function=embedding_function,
       top_n=k_reranker,
       reranking_function=reranking_function,
       r_score=r,
   )
   compression_retriever = ContextualCompressionRetriever(
       base_compressor=compressor, 
       base_retriever=ensemble_retriever
   )
   result = await compression_retriever.ainvoke(query)
   ```

5. **Post-procesamiento**:
   - Ordenar por score (descendente)
   - Limitar a `min(k, k_reranker)` si `k < k_reranker`
   - Añadir scores a metadata

6. **Retorno**:
   ```python
   {
       "distances": [[score1, score2, ...]],
       "documents": [[doc1, doc2, ...]],
       "metadatas": [[meta1, meta2, ...]]
   }
   ```

### 1.6 Búsqueda Multi-Colección

**Función**: `query_collection_with_hybrid_search()`

Orquesta búsqueda híbrida en múltiples colecciones en paralelo:

```python
async def query_collection_with_hybrid_search(
    collection_names: list[str],
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
    enable_enriched_texts: bool = False,
) -> dict:
    # Obtener todos los documentos de las colecciones
    get_results = await asyncio.gather(
        *[VECTOR_DB_CLIENT.get(name) for name in collection_names]
    )
    
    # Búsqueda híbrida en paralelo por colección
    results = await asyncio.gather(
        *[query_doc_with_hybrid_search(
            collection_name=name,
            collection_result=result,
            query=query,
            # ... otros parámetros
        ) for name, result in zip(collection_names, get_results)]
    )
    
    # Fusión de resultados de todas las colecciones
    return merge_get_results(results)
```

**Características**:
- Ejecución **paralela** con `asyncio.gather`
- Cada colección retorna hasta `TOP_K` documentos
- Fusión final de todas las colecciones
- Deduplicación si un documento aparece en múltiples colecciones

### 1.7 Bases de Datos Vectoriales Soportadas

| Backend | Multitenancy | Filtrado Metadata | Búsqueda Híbrida Nativa |
|---------|--------------|-------------------|-------------------------|
| **ChromaDB** | ✓ | ✓ | ❌ (implementada en app) |
| **Qdrant** | ✓ | ✓ | ✓ (fusion + rerank interno) |
| **Milvus** | ✓ | ✓ | ✓ (keyword + dense vector) |
| **pgvector** | ✓ | ✓ | ❌ (implementada en app) |
| **Pinecone** | ✓ | ✓ | ❌ (implementada en app) |
| **Weaviate** | ✓ | ✓ | ✓ (BM25 + vector fusion) |
| **Elasticsearch** | ✓ | ✓ | ✓ (BM25 + kNN) |
| **OpenSearch** | ✓ | ✓ | ✓ (lexical + neural) |
| **Oracle 23ai** | ✓ | ✓ | ❌ (implementada en app) |
| **OpenGauss** | ❌ | ✓ | ❌ (implementada en app) |

**Notas**:
- La implementación de Open WebUI es **agnóstica al backend**
- Incluso con backends con híbrida nativa, usa su propia implementación para consistencia
- Ventaja: Mismo comportamiento independiente de la base de datos

### 1.8 Ejemplo de Flujo Completo

**Escenario**: Usuario pregunta "¿Cómo instalar Open WebUI con Docker?"

1. **Configuración Activa**:
   - `ENABLE_RAG_HYBRID_SEARCH=True`
   - `RAG_HYBRID_BM25_WEIGHT=0.5`
   - `TOP_K=10`
   - `TOP_K_RERANKER=3`
   - `RELEVANCE_THRESHOLD=0.0`

2. **Procesamiento del Query**:
   - Query: "¿Cómo instalar Open WebUI con Docker?"

3. **BM25 Retriever**:
   - Tokeniza: ["como", "instalar", "open", "webui", "docker"]
   - Calcula BM25 scores para cada documento
   - TOP 10 documentos por keyword matching:
     ```
     Doc1: "Docker installation guide..." (BM25: 12.5)
     Doc5: "Installing with Docker Compose..." (BM25: 11.8)
     Doc8: "Docker setup for Open WebUI..." (BM25: 10.3)
     ...
     ```

4. **Vector Retriever**:
   - Embedding del query: `[0.23, -0.45, 0.67, ...]` (768/1024 dims)
   - Búsqueda de vecinos más cercanos (cosine similarity)
   - TOP 10 documentos por similitud semántica:
     ```
     Doc2: "Setting up the application in containers..." (sim: 0.89)
     Doc5: "Installing with Docker Compose..." (sim: 0.87)
     Doc12: "Deployment using containerization..." (sim: 0.82)
     ...
     ```

5. **Ensemble Fusion (RRF)**:
   - Doc1: `0.5*(1/(1+60)) + 0.5*(1/(6+60))` = 0.011
   - Doc2: `0.5*(1/(8+60)) + 0.5*(1/(1+60))` = 0.011
   - Doc5: `0.5*(1/(2+60)) + 0.5*(1/(2+60))` = 0.016 ⭐ (aparece en ambos TOP 2)
   - ...
   - Resultado: 15-20 documentos únicos fusionados

6. **Reranking (Cross-Encoder)**:
   - Modelo: `jinaai/jina-reranker-v2`
   - Re-score de 15 candidatos:
     ```
     Doc5: 0.94 ⭐
     Doc8: 0.87 ⭐
     Doc2: 0.82 ⭐
     Doc1: 0.76
     ...
     ```

7. **Resultado Final**:
   - TOP 3 documentos (k_reranker=3):
     ```json
     {
       "documents": [
         ["Installing with Docker Compose...", 
          "Docker setup for Open WebUI...", 
          "Setting up the application in containers..."]
       ],
       "metadatas": [
         [{"source": "docs/install.md", "score": 0.94}, 
          {"source": "README.md", "score": 0.87}, 
          {"source": "docs/deploy.md", "score": 0.82}]
       ],
       "distances": [[0.94, 0.87, 0.82]]
     }
     ```

---

## 2. Guía de Onboarding: Entendiendo la Búsqueda Híbrida

### 2.1 ¿Qué es la Búsqueda Híbrida?

Imagina que estás buscando un libro en una biblioteca gigante. Tienes dos estrategias:

1. **Búsqueda por Palabras Clave (BM25)**: 
   - Como usar el catálogo de la biblioteca
   - Buscas libros que contengan exactamente las palabras que escribiste
   - Ejemplo: Buscas "Docker instalación" → encuentra libros con esas palabras exactas
   - **Pros**: Encuentra coincidencias exactas, rápido
   - **Contras**: No entiende sinónimos o conceptos similares

2. **Búsqueda Semántica (Vectorial)**:
   - Como pedirle ayuda a un bibliotecario experto
   - Busca documentos que hablen de conceptos similares, aunque usen palabras diferentes
   - Ejemplo: Buscas "configurar contenedores" → encuentra "instalar Docker", "setup containerización"
   - **Pros**: Entiende el significado, encuentra información relacionada
   - **Contras**: A veces "demasiado creativo", puede traer resultados menos precisos

**La Búsqueda Híbrida** combina ambas estrategias para obtener lo mejor de ambos mundos.

### 2.2 ¿Por Qué Necesitamos Búsqueda Híbrida?

**Problema 1: Búsqueda por Palabras Clave Sola**
```
Query: "Cómo configurar autenticación OAuth2"
BM25 solo encuentra: Documentos con "OAuth2", "configurar", "autenticación"
Pierde: Documentos que hablan de "integración SSO", "login social", "autenticación externa"
```

**Problema 2: Búsqueda Semántica Sola**
```
Query: "Error 404 en API /v1/users"
Vector encuentra: Documentos generales sobre errores HTTP, APIs REST
Pierde: Documentos que mencionan específicamente "/v1/users" o "404"
```

**Solución: Híbrida**
```
Query: "Error 404 en API /v1/users"
BM25 trae: Documentos con path exacto "/v1/users"
Vector trae: Documentos sobre troubleshooting de APIs, errores HTTP
Resultado: Documentación específica del endpoint + guías de debugging
```

### 2.3 Conceptos Clave Simplificados

#### 2.3.1 Embeddings (Vectores)

**¿Qué son?**
- Representación numérica del significado de un texto
- Como "coordenadas GPS" del significado en un espacio multidimensional
- Textos similares están cerca en este espacio

**Ejemplo Visual** (simplificado a 2D):
```
Espacio de Embeddings:

    "instalar Docker"  ●
                      
    "configurar contenedor"  ●
    
                        
                          ● "receta de cocina"
                          
    "setup Docker Compose" ●
```

**Proceso de Embedding**:
```
Texto: "Instalar Open WebUI"
         ↓ (Modelo de embedding)
Vector: [0.23, -0.45, 0.67, 0.12, ..., -0.34]
         (768 o 1024 números)
```

#### 2.3.2 BM25 (Keyword Matching)

**Fórmula Simplificada**:
```
Score(documento, query) = Σ IDF(palabra) × TF(palabra, documento)

Donde:
- TF = Cuántas veces aparece la palabra en el documento (normalizado)
- IDF = Qué tan rara es la palabra en toda la colección
```

**Ejemplo Práctico**:
```
Query: "Docker instalación Ubuntu"

Documento A: "Guía de instalación de Docker en Ubuntu 22.04..."
- "Docker": TF=5, IDF=2.3 → 11.5
- "instalación": TF=3, IDF=1.8 → 5.4
- "Ubuntu": TF=4, IDF=2.1 → 8.4
→ Score total: 25.3

Documento B: "Tutorial de configuración de contenedores..."
- "Docker": TF=1, IDF=2.3 → 2.3
- "instalación": TF=0, IDF=1.8 → 0
- "Ubuntu": TF=0, IDF=2.1 → 0
→ Score total: 2.3

Documento A gana (25.3 > 2.3)
```

#### 2.3.3 RRF (Reciprocal Rank Fusion)

**Problema**: BM25 y Vector scores están en escalas diferentes
- BM25: scores de 0 a 50+ (sin límite)
- Vector: similitud coseno de 0 a 1

**Solución RRF**: Fusionar por **ranking** (posición), no por score absoluto

**Ejemplo**:
```
Query: "instalar Docker"

Ranking BM25:
1. Doc A (score: 25.3)
2. Doc C (score: 18.7)
3. Doc B (score: 12.1)

Ranking Vector:
1. Doc B (similarity: 0.92)
2. Doc A (similarity: 0.88)
3. Doc D (similarity: 0.81)

RRF Fusion (peso 50/50):
Doc A: 0.5×(1/(1+60)) + 0.5×(1/(2+60)) = 0.012
Doc B: 0.5×(1/(3+60)) + 0.5×(1/(1+60)) = 0.012
Doc C: 0.5×(1/(2+60)) + 0.5×(1/(∞+60)) = 0.008
Doc D: 0.5×(1/(∞+60)) + 0.5×(1/(3+60)) = 0.008

Ranking Final:
1. Doc A (0.012) ← aparece en TOP 2 de ambos
2. Doc B (0.012) ← aparece en TOP 3 de ambos
3. Doc C (0.008)
4. Doc D (0.008)
```

#### 2.3.4 Reranking (Re-ordenamiento)

**¿Por qué reordenar?**
- BM25 + Vector traen buenos candidatos, pero no perfecto
- Un modelo más sofisticado (Cross-Encoder) re-evalúa la relevancia

**Diferencia Bi-Encoder vs Cross-Encoder**:

```
BI-ENCODER (Usado en Vector Search):
Query → Embedding A  ─┐
                       ├─→ Similitud (coseno, dot product)
Documento → Embedding B─┘
(Rápido: embeddings pre-calculados)

CROSS-ENCODER (Usado en Reranking):
[Query + Documento] → Modelo → Score directo
(Lento pero preciso: procesa query+doc juntos)
```

**Ejemplo**:
```
Candidatos del Ensemble (15 documentos):
Doc1, Doc2, Doc3, ..., Doc15

Cross-Encoder re-score:
Doc5: 0.94 ← MÁS relevante
Doc8: 0.87
Doc2: 0.82
Doc1: 0.76
...
Doc12: 0.31 ← menos relevante

TOP_K_RERANKER = 3 → Retorna Doc5, Doc8, Doc2
```

### 2.4 Flujo de Usuario Completo (Paso a Paso)

#### Escenario: Usuario Nuevo Quiere Entender RAG

**Paso 1: Usuario Sube Documentos**
```
Usuario → Upload PDF: "RAG_Paper.pdf"
   ↓
Open WebUI:
1. Extrae texto del PDF
2. Divide en chunks de 1000 caracteres
3. Genera embeddings de cada chunk
4. Guarda en Vector DB (ChromaDB/Qdrant/etc.)
```

**Paso 2: Usuario Hace una Pregunta**
```
Usuario → Chat: "¿Qué es RAG y cómo funciona?"
   ↓
Open WebUI activa búsqueda híbrida si está habilitada
```

**Paso 3: BM25 Retriever Busca por Keywords**
```
Tokeniza: ["rag", "funciona"]
   ↓
Calcula BM25 scores en TODOS los chunks
   ↓
TOP 10 chunks con mayor keyword match:
- Chunk 42: "RAG (Retrieval Augmented Generation) es..."
- Chunk 15: "El funcionamiento de RAG se basa en..."
- Chunk 88: "RAG combina búsqueda con generación..."
```

**Paso 4: Vector Retriever Busca por Significado**
```
Embedding del query: [0.23, -0.45, ...]
   ↓
Búsqueda de vecinos más cercanos en Vector DB
   ↓
TOP 10 chunks con mayor similitud semántica:
- Chunk 42: similarity 0.91
- Chunk 103: similarity 0.89 (habla de "recuperación aumentada")
- Chunk 25: similarity 0.87 (explica "generación con contexto")
```

**Paso 5: Ensemble Fusiona Resultados**
```
RRF combina rankings de BM25 y Vector
   ↓
Pesos: 50% BM25, 50% Vector
   ↓
Resultado: ~15 chunks únicos fusionados
```

**Paso 6: Reranking Refina**
```
Cross-Encoder evalúa cada par (query, chunk)
   ↓
Scores:
- Chunk 42: 0.94 ⭐
- Chunk 15: 0.89 ⭐
- Chunk 103: 0.85 ⭐
- Chunk 88: 0.79
   ↓
TOP_K_RERANKER=3 → Retorna los 3 mejores
```

**Paso 7: LLM Genera Respuesta**
```
Contexto (TOP 3 chunks) → LLM
   ↓
LLM genera respuesta basándose en el contexto:
"RAG (Retrieval Augmented Generation) es una técnica que combina 
búsqueda de información con generación de texto. Funciona en dos fases:
1. Recuperación: busca documentos relevantes en una base de conocimientos
2. Generación: usa esos documentos como contexto para generar la respuesta
..."
```

### 2.5 Configuraciones para Casos de Uso Comunes

#### Caso 1: Documentación Técnica (Alta Precisión en Keywords)
```bash
ENABLE_RAG_HYBRID_SEARCH=true
RAG_HYBRID_BM25_WEIGHT=0.7  # 70% BM25, 30% Vector
TOP_K=15
TOP_K_RERANKER=5
RELEVANCE_THRESHOLD=0.5
```
**Por qué**: Términos técnicos exactos (nombres de funciones, comandos) son cruciales

#### Caso 2: Preguntas Conceptuales (Alta Comprensión Semántica)
```bash
ENABLE_RAG_HYBRID_SEARCH=true
RAG_HYBRID_BM25_WEIGHT=0.3  # 30% BM25, 70% Vector
TOP_K=20
TOP_K_RERANKER=3
RELEVANCE_THRESHOLD=0.3
```
**Por qué**: Importa más el significado que palabras exactas

#### Caso 3: Búsqueda de Código (Solo Keywords)
```bash
ENABLE_RAG_HYBRID_SEARCH=true
RAG_HYBRID_BM25_WEIGHT=1.0  # 100% BM25
TOP_K=10
TOP_K_RERANKER=5
RELEVANCE_THRESHOLD=0.0
```
**Por qué**: Código requiere coincidencias exactas de funciones/variables

#### Caso 4: Base de Conocimiento General (Balanceado)
```bash
ENABLE_RAG_HYBRID_SEARCH=true
RAG_HYBRID_BM25_WEIGHT=0.5  # 50/50 (default)
TOP_K=10
TOP_K_RERANKER=3
RELEVANCE_THRESHOLD=0.0
```
**Por qué**: Balance entre precisión keyword y comprensión semántica

### 2.6 Preguntas Frecuentes (FAQ)

**Q1: ¿Cuándo se activa la búsqueda híbrida?**
- Cuando `ENABLE_RAG_HYBRID_SEARCH=True` (default)
- Se puede desactivar por colección vía API
- Si está desactivada, solo usa Vector Search

**Q2: ¿Qué pasa si pongo RAG_HYBRID_BM25_WEIGHT=0?**
- Solo búsqueda vectorial (semántica)
- BM25 no se ejecuta
- Útil si el modelo de embeddings es muy bueno

**Q3: ¿El reranking es obligatorio?**
- No, es opcional
- Si no hay `reranking_function`, usa cosine similarity
- Pero reranking mejora calidad significativamente

**Q4: ¿Cuál es la diferencia entre TOP_K y TOP_K_RERANKER?**
- `TOP_K`: Documentos iniciales de cada retriever (BM25 y Vector)
- `TOP_K_RERANKER`: Documentos finales tras reranking
- Típico: TOP_K=10, TOP_K_RERANKER=3
- Reranking evalúa 10-20 candidatos, retorna los mejores 3

**Q5: ¿Cómo afecta RELEVANCE_THRESHOLD?**
- Filtra documentos con score < threshold
- 0.0 = incluye todos
- 0.5 = solo medianamente relevantes
- 0.8 = solo muy relevantes (puede retornar 0 resultados)

**Q6: ¿Puedo usar búsqueda híbrida con cualquier Vector DB?**
- Sí, la implementación es agnóstica al backend
- Funciona con ChromaDB, Qdrant, Milvus, pgvector, etc.
- Incluso si el DB tiene híbrida nativa, Open WebUI usa su propia implementación

**Q7: ¿Qué pasa si no tengo documentos suficientes?**
- BM25 necesita al menos algunos documentos para funcionar
- Con <5 documentos, recomendado solo Vector Search
- Con 5-100 documentos, híbrida funciona bien
- Con 100+ documentos, híbrida brilla

---

## 3. Comparativa: Búsqueda Híbrida In-Process vs Microservicio Dedicado

### 3.1 Arquitecturas Comparadas

#### 3.1.1 Implementación Actual: In-Process (Monolítica)

```
┌───────────────────────────────────────────────────────────┐
│                    OPEN WEBUI (Python)                     │
├───────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   FastAPI   │  │  BM25        │  │  Vector Search  │  │
│  │   Routers   │──│  Retriever   │──│  Retriever      │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
│         │                │                     │          │
│         └────────────────┴─────────────────────┘          │
│                          │                                │
│                   ┌──────▼───────┐                        │
│                   │   Ensemble   │                        │
│                   │   + Rerank   │                        │
│                   └──────┬───────┘                        │
│                          │                                │
└──────────────────────────┼────────────────────────────────┘
                           │
                           v
              ┌────────────────────────┐
              │   Vector DB            │
              │  (ChromaDB/Qdrant/etc.)│
              └────────────────────────┘
```

**Características**:
- Todo en un solo proceso Python
- BM25, Vector, Ensemble, Rerank en memoria
- Comunicación directa (sin red)

#### 3.1.2 Alternativa: Microservicio Dedicado

```
┌────────────────────┐         ┌──────────────────────────────┐
│   OPEN WEBUI       │  HTTP/  │  SEARCH MICROSERVICE         │
│   (Python)         │  gRPC   │  (Python/Go/Rust)            │
├────────────────────┤ ──────> ├──────────────────────────────┤
│ • LLM routing      │         │  ┌────────────────────────┐  │
│ • UI/API           │         │  │  Search Orchestrator   │  │
│ • Auth             │         │  ├────────────────────────┤  │
│ • File upload      │         │  │ • BM25 Engine          │  │
└────────────────────┘         │  │ • Vector Engine        │  │
                               │  │ • Ensemble Fusion      │  │
                               │  │ • Reranking Service    │  │
                               │  │ • Cache Layer          │  │
                               │  └───────┬────────────────┘  │
                               │          │                   │
                               └──────────┼───────────────────┘
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                              v                       v
                    ┌──────────────┐      ┌─────────────────┐
                    │  Vector DB   │      │  Redis Cache    │
                    │  (dedicated) │      │  (results)      │
                    └──────────────┘      └─────────────────┘
```

**Características**:
- Servicio independiente
- API REST/gRPC
- Escalado independiente
- Posible implementación en lenguaje optimizado (Rust/Go)

### 3.2 Tabla Comparativa Detallada

| Aspecto | In-Process (Actual) | Microservicio Dedicado |
|---------|---------------------|------------------------|
| **DESARROLLO** | | |
| Complejidad de desarrollo | ⭐⭐ (Baja) | ⭐⭐⭐⭐ (Alta) |
| Tiempo de implementación | Rápido (ya existe) | Lento (semanas/meses) |
| Debugging | Fácil (un solo proceso) | Complejo (distribuido) |
| Testing | Simple (unit tests) | Complejo (integration/e2e) |
| **OPERACIÓN** | | |
| Complejidad de despliegue | ⭐⭐ (Baja) | ⭐⭐⭐⭐⭐ (Muy alta) |
| Número de servicios | 1 (+ Vector DB) | 3+ (App + Search + Vector DB + Cache) |
| Configuración | Variables de entorno | Variables + Networking + Service discovery |
| Monitoreo | 1 proceso | Múltiples servicios + trazabilidad distribuida |
| **RENDIMIENTO** | | |
| Latencia base | ~50-200ms | ~100-400ms (+ overhead red) |
| Overhead de red | 0ms | 10-50ms (LAN), 50-200ms (WAN) |
| Throughput (búsquedas/seg) | ~10-50 | ~100-1000+ (con escalado) |
| Caché | En memoria del proceso | Redis/Memcached dedicado |
| **ESCALABILIDAD** | | |
| Escalado horizontal | ❌ Difícil (réplicas completas) | ✅ Fácil (solo search service) |
| Escalado vertical | ✅ Aumentar CPU/RAM del pod | ✅ Ajustar recursos solo del search |
| Cuellos de botella | Todo el app comparte recursos | Search aislado, no afecta app |
| Costo de escalar | Alto (réplica completa) | Bajo (solo componente search) |
| **MANTENIBILIDAD** | | |
| Acoplamiento | Alto (código integrado) | Bajo (interfaz API bien definida) |
| Actualización de algoritmos | Requiere redeploy completo | Solo redeploy del search service |
| Testing de búsqueda | Acoplado al app | Independiente |
| Rollback | Todo o nada | Solo search service |
| **FLEXIBILIDAD** | | |
| Cambio de algoritmo | Moderado (código Python) | Fácil (API agnóstica) |
| Múltiples backends | 1 configuración por instancia | N configuraciones por tenant |
| A/B Testing | Difícil | Fácil (routing en API Gateway) |
| Lenguaje de implementación | Python (limitado) | Cualquiera (Rust/Go/C++ para speed) |
| **RESILIENCIA** | | |
| Single Point of Failure | ✅ App cae → todo cae | ⚠️ Múltiples puntos de fallo |
| Aislamiento de fallos | ❌ Error en búsqueda → app down | ✅ Error en búsqueda → app sigue |
| Circuit breaking | No aplica | Necesario implementar |
| Retry logic | Simple (en memoria) | Complejo (timeouts, retries) |
| **SEGURIDAD** | | |
| Superficie de ataque | Pequeña (1 servicio) | Amplia (múltiples servicios) |
| Autenticación | Session-based | JWT/OAuth + service-to-service |
| Rate limiting | Global | Por servicio + global |
| Network isolation | No aplica | VPC/Service mesh necesario |
| **COSTOS** | | |
| Infraestructura | $ (1 app instance) | $$$ (App + Search + Cache + LB) |
| Desarrollo inicial | $ (ya existe) | $$$$ (meses de dev) |
| Mantenimiento | $$ | $$$$ (múltiples servicios) |
| Networking | Gratis (local) | $$ (data transfer entre servicios) |
| **OBSERVABILIDAD** | | |
| Logging | Simple (un log stream) | Complejo (agregación necesaria) |
| Métricas | Prometheus simple | Prometheus + Grafana + Jaeger |
| Distributed tracing | No necesario | Esencial (OpenTelemetry) |
| Debugging producción | Moderado | Difícil (multiple hops) |

### 3.3 Análisis de Casos de Uso

#### Caso 1: Startup/Proyecto Pequeño (<10K usuarios)

**Recomendación: In-Process ✅**

**Razones**:
- Recursos limitados (equipo, infraestructura, tiempo)
- Complejidad operacional mínima
- Latencia no crítica para este volumen
- Costos controlados

**Métricas**:
- Búsquedas/día: <100K
- Latencia aceptable: <500ms
- Disponibilidad objetivo: 99%

#### Caso 2: Empresa Mediana (10K-100K usuarios)

**Recomendación: In-Process ⚠️ → Microservicio si creces rápido**

**Razones**:
- In-process aún viable con escalado vertical
- Considera microservicio si:
  - Latencia >500ms consistentemente
  - CPU de búsqueda >50% del total
  - Necesitas A/B testing de algoritmos

**Métricas**:
- Búsquedas/día: 100K-1M
- Latencia aceptable: <300ms
- Disponibilidad objetivo: 99.5%

**Punto de inflexión**:
```
Si [CPU búsqueda] > 50% de [CPU total]
    → Considera microservicio
    
Si [p95 latency búsqueda] > 500ms
    → Optimiza primero, luego microservicio
    
Si necesitas [escalar búsqueda] 3x más que [escalar app]
    → Microservicio justificado
```

#### Caso 3: Empresa Grande (>100K usuarios)

**Recomendación: Microservicio ✅**

**Razones**:
- Escalado horizontal necesario
- Optimizaciones específicas (Rust/Go para BM25)
- A/B testing de algoritmos
- SLAs estrictos de latencia
- Equipos especializados

**Métricas**:
- Búsquedas/día: >1M
- Latencia requerida: <100ms (p99)
- Disponibilidad objetivo: 99.9%+

**Arquitectura Recomendada**:
```
┌────────────────────┐
│   Load Balancer    │
└─────────┬──────────┘
          │
    ┌─────┴─────┐
    │           │
┌───▼───┐   ┌──▼────┐
│ App 1 │   │ App N │
└───┬───┘   └──┬────┘
    │          │
    └────┬─────┘
         │ gRPC/HTTP
    ┌────▼──────────────────┐
    │  Search API Gateway   │
    │  (rate limit, cache)  │
    └────┬──────────────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│Search1│ │SearchN│ (auto-scale)
└───┬───┘ └──┬────┘
    │        │
    └───┬────┘
        │
┌───────▼────────┐
│   Vector DB    │
│   (clustered)  │
└────────────────┘
```

#### Caso 4: Multi-tenant SaaS

**Recomendación: Microservicio ✅ (Imprescindible)**

**Razones**:
- Aislamiento por tenant
- Configuraciones personalizadas por cliente
- Facturación por uso de búsqueda
- Rate limiting granular

**Arquitectura Específica**:
```
Tenant A → Search Service (config A, limits A)
Tenant B → Search Service (config B, limits B)
Tenant C → Search Service (config C, limits C)

Cada tenant puede tener:
- Diferentes pesos BM25/Vector
- Diferentes modelos de embedding
- Diferentes umbrales de relevancia
- Diferentes límites de tasa
```

### 3.4 Análisis de Costos

#### Escenario: 50K usuarios activos, 500K búsquedas/día

**In-Process:**
```
Infraestructura:
- 3 instancias app (4 CPU, 8GB RAM): $300/mes
- Vector DB managed (Qdrant Cloud): $200/mes
- Total: $500/mes

Desarrollo:
- Mantenimiento: 5 horas/mes × $100/hora = $500/mes

Total mensual: $1,000
```

**Microservicio:**
```
Infraestructura:
- 3 instancias app (2 CPU, 4GB RAM): $150/mes
- 5 instancias search (4 CPU, 8GB RAM): $500/mes
- Vector DB managed: $200/mes
- Redis cache: $50/mes
- Load Balancer: $50/mes
- Networking: $50/mes
- Total: $1,000/mes

Desarrollo:
- Implementación inicial: 160 horas × $100/hora = $16,000 (one-time)
- Mantenimiento: 20 horas/mes × $100/hora = $2,000/mes

Total mensual: $3,000 (+ $16K inicial)
Año 1: $52,000
```

**ROI del Microservicio**:
- Solo justificado si:
  - Mejora latencia genera >$2K/mes valor (retención, conversión)
  - Escalabilidad permite crecer sin replantear arquitectura
  - Equipo especializado en búsqueda justifica inversión

### 3.5 Estrategia de Migración Gradual

Si decides migrar de In-Process a Microservicio:

#### Fase 1: Extracción (Mes 1-2)
```
1. Crear microservicio con API idéntica al código actual
2. Deployar en paralelo (sin usar)
3. Test suite completo
```

#### Fase 2: Canary Testing (Mes 2-3)
```
1. Ruta 1% del tráfico al microservicio
2. Comparar resultados (A/B testing)
3. Monitorear latencia, errores
4. Incrementar gradualmente: 1% → 5% → 10% → 25%
```

#### Fase 3: Feature Flags (Mes 3-4)
```
1. Feature flag por usuario/tenant
2. Usuarios beta en microservicio
3. Rollback fácil si problemas
```

#### Fase 4: Migración Completa (Mes 4-5)
```
1. 100% tráfico al microservicio
2. Deprecar código in-process
3. Optimizaciones específicas del microservicio
```

#### Fase 5: Optimización (Mes 5+)
```
1. Reescribir componentes críticos en Rust/Go
2. Caché agresivo
3. Sharding por tenant
4. Auto-scaling avanzado
```

### 3.6 Recomendaciones Finales

#### Mantén In-Process si:
- ✅ Volumen <100K búsquedas/día
- ✅ Equipo pequeño (<5 devs)
- ✅ Latencia <500ms aceptable
- ✅ Presupuesto limitado
- ✅ Time-to-market es prioridad

#### Considera Microservicio si:
- ⚠️ Búsqueda consume >50% CPU del app
- ⚠️ Necesitas escalar búsqueda independientemente
- ⚠️ Múltiples equipos trabajan en el sistema
- ⚠️ Quieres A/B testing de algoritmos
- ⚠️ Latencia es crítica (<100ms requerida)

#### Migra a Microservicio si:
- ❗ Volumen >1M búsquedas/día
- ❗ SLAs estrictos (99.9%+ uptime)
- ❗ Multi-tenant con aislamiento requerido
- ❗ Optimizaciones de lenguaje necesarias (Rust/Go)
- ❗ Equipo dedicado a búsqueda disponible

### 3.7 Alternativas Híbridas (Lo Mejor de Ambos Mundos)

#### Opción 1: Search Sidecar Pattern
```
┌───────────────────────────────┐
│         POD                    │
│  ┌──────────┐  ┌────────────┐ │
│  │          │  │   Search   │ │
│  │   App    │──│  Sidecar   │ │
│  │ Container│  │  Container │ │
│  └──────────┘  └─────┬──────┘ │
└────────────────────┼──────────┘
                      │ localhost
                      v
              ┌────────────┐
              │ Vector DB  │
              └────────────┘
```
**Ventajas**:
- Latencia ultra-baja (localhost)
- Aislamiento de proceso
- Escalado conjunto (app + search)
- Actualización independiente de containers

#### Opción 2: Cached Microservice
```
┌──────────┐     ┌──────────────┐
│   App    │────>│  Redis Cache │
└────┬─────┘     └──────┬───────┘
     │ cache miss        │
     └───────────────────┘
            │
            v
    ┌──────────────┐
    │   Search MS  │
    └──────────────┘
```
**Ventajas**:
- 90%+ queries desde cache (latencia <10ms)
- Microservicio solo para queries complejos
- Costo reducido

#### Opción 3: Embedding Separation Only
```
┌────────────────────┐
│       App          │
│  • BM25 (in-proc)  │
│  • Ensemble        │
│  • Rerank          │
└────────┬───────────┘
         │
         v
┌────────────────────┐
│  Vector Search MS  │ ← Solo este separado
│  (GPU optimizado)  │
└────────────────────┘
```
**Ventajas**:
- BM25 rápido en CPU local
- Vector search en GPU dedicado
- Complejidad moderada

---

## Conclusión

La **búsqueda híbrida in-process** de Open WebUI es una solución sólida, bien diseñada y suficiente para la gran mayoría de casos de uso. Solo considera migrar a microservicio cuando tengas métricas claras que lo justifiquen (volumen, latencia, escalabilidad).

### Resumen Ejecutivo

| Aspecto | In-Process | Microservicio |
|---------|-----------|---------------|
| **Mejor para** | Startups, proyectos pequeños/medianos | Empresas grandes, SaaS multi-tenant |
| **Complejidad** | Baja ⭐⭐ | Alta ⭐⭐⭐⭐⭐ |
| **Costo** | Bajo $ | Alto $$$ |
| **Time-to-market** | Rápido ✅ | Lento ⏳ |
| **Escalabilidad** | Vertical (limitada) | Horizontal (ilimitada) |
| **Latencia típica** | 100-300ms | 50-150ms (con optimizaciones) |
| **Mantenimiento** | Simple | Complejo |

**Regla de oro**: No migres a microservicio hasta que **demuestres con datos** que el in-process es un cuello de botella real. La sobre-ingeniería prematura es costosa y distrae del producto.

---

## Anexo: Referencias Técnicas

### Modelos de Reranking Recomendados

| Modelo | Idioma | Latencia | Calidad | Uso Recomendado |
|--------|--------|----------|---------|-----------------|
| `jinaai/jina-reranker-v2-base-multilingual` | Multi | 50ms | Alta | General purpose |
| `BAAI/bge-reranker-v2-m3` | Multi | 40ms | Alta | Documentos largos |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | EN | 30ms | Media | Baja latencia |
| `jinaai/jina-colbert-v2` | Multi | 80ms | Muy Alta | Máxima precisión |

### Embeddings Recomendados

| Modelo | Idioma | Dimensión | Rendimiento |
|--------|--------|-----------|-------------|
| `jinaai/jina-embeddings-v3` | Multi | 1024 | Excelente |
| `BAAI/bge-m3` | Multi | 1024 | Excelente |
| `intfloat/multilingual-e5-large` | Multi | 1024 | Muy bueno |
| `sentence-transformers/all-MiniLM-L6-v2` | EN | 384 | Bueno (rápido) |

### Bases de Datos Vectoriales Recomendadas

| Base de Datos | Escalabilidad | Facilidad | Costo | Mejor Para |
|---------------|---------------|-----------|-------|-----------|
| **ChromaDB** | Media | Alta | Gratis | Dev/Testing |
| **Qdrant** | Alta | Alta | $$ | Producción pequeña/mediana |
| **Milvus** | Muy Alta | Media | $$$ | Producción grande |
| **pgvector** | Media | Alta | $ | Si ya usas PostgreSQL |
| **Pinecone** | Alta | Muy Alta | $$$ | SaaS, sin gestión |

---

**Documento creado**: 2026-01-27  
**Versión**: 1.0  
**Autor**: Open WebUI Documentation Team  
**Licencia**: MIT (igual que el proyecto)
