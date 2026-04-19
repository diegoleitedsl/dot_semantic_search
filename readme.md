# Busca Semântica com Embeddings e FAISS

Sistema de busca semântica de documentos usando embeddings e vector store.
Sem frameworks de orquestração — só o necessário.

---

## Tecnologias

| Biblioteca            | Papel                                     |
| --------------------- | ----------------------------------------- |
| sentence-transformers | Geração de embeddings (vetores numéricos) |
| faiss-cpu             | Vector store e busca por similaridade L2  |
| numpy                 | Manipulação dos vetores                   |
| Python stdlib         | Tudo o mais                               |

---

## Estrutura

text
Copiar

dot_semantic/
├── app/
│ ├── documents.py # corpus de documentos
│ └── semantic_search.py # lógica de embeddings e busca
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

---

## Como funciona

### 1. Documentos

Definidos em `app/documents.py` como lista de strings.
Podem ser artigos, posts, trechos de documentação — qualquer corpus de texto.

### 2. Embeddings

O modelo `all-MiniLM-L6-v2` converte cada documento em um vetor de 384 dimensões.
Documentos semanticamente próximos ficam próximos no espaço vetorial.


vectors = model.encode(docs, convert_to_numpy=True)

### 3. Vector Store (FAISS)

Os vetores são adicionados a um índice `IndexFlatL2`.
Esse índice calcula distância euclidiana (L2) — quanto menor, mais relevante.


index = faiss.IndexFlatL2(dim)
index.add(vectors)

### 4. Busca semântica

A query é convertida no mesmo espaço vetorial e comparada contra o índice:

distances, indices = index.search(query_vec, top_k)

O FAISS retorna os `top_k` documentos com menor distância.

---

## Execução com Docker


docker compose build --no-cache
docker compose run --rm semantic

### Saída esperada

```
Carregando modelo de embeddings...
Indexando 12 documentos...
Índice criado — 12 vetores, dimensão 384

=== Demonstração ===

Consulta : 'Como percorrer elementos de uma lista?'
────────────────────────────────────────────────────────────
  1. [dist=0.4821] O laço for percorre qualquer iterável: listas, tuplas, strings e ranges.
  2. [dist=0.8934] Listas em Python são coleções ordenadas e mutáveis: ex. numeros = [1, 2, 3].
  3. [dist=1.1023] List comprehensions criam listas de forma concisa: [x*2 for x in range(10)].

Consulta : 'Quero usar Python para inteligência artificial'
────────────────────────────────────────────────────────────
  1. [dist=0.5103] Python é amplamente usado em Machine Learning com scikit-learn e TensorFlow.
  2. [dist=0.9812] Python é uma linguagem interpretada de alto nível com tipagem dinâmica.
  3. [dist=1.2341] Funções são definidas com def e podem retornar qualquer tipo de dado.

Consulta : 'Como tratar erros no código?'
────────────────────────────────────────────────────────────
  1. [dist=0.6210] Exceções são tratadas com try, except, else e finally.
  2. [dist=1.0432] Funções são definidas com def e podem retornar qualquer tipo de dado.
  3. [dist=1.3211] O módulo os permite interagir com o sistema operacional e variáveis de ambiente.

=== Modo interativo (digite 'sair' para encerrar) ===

Consulta:
```

### Encerrar

Digite `sair`, `exit` ou `quit` no prompt `Consulta:`.

---

