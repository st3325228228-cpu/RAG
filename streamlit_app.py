"""
Streamlit + Groq API - 8種 RAG 策略 PDF 問答系統
執行: streamlit run app.py
"""

import os
import logging
import re
import tempfile
from typing import Optional

import streamlit as st
import numpy as np
import faiss
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ── 日誌設定 ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 常數 ─────────────────────────────────────────────────
DEFAULT_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SMALL_CHUNK_SIZE = 300
SMALL_CHUNK_OVERLAP = 50


# ══════════════════════════════════════════════════════════
#  快取：Embedding 模型只載入一次
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄 首次載入 Embedding 模型中…")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ══════════════════════════════════════════════════════════
#  RAG 引擎核心類別
# ══════════════════════════════════════════════════════════
class MultiStrategyRAG:
    """支援 8 種 RAG 策略的 PDF 問答引擎"""

    def __init__(self):
        self.client: Optional[Groq] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.chunks: list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.pdf_loaded = False

    # ── API Key ───────────────────────────────────────────
    def set_api_key(self, api_key: str) -> tuple[bool, str]:
        api_key = api_key.strip()
        if not api_key:
            return False, "❌ 請輸入有效的 API Key"
        try:
            self.client = Groq(api_key=api_key)
            self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            return True, "✅ API Key 驗證成功！"
        except Exception as e:
            self.client = None
            return False, f"❌ API Key 無效: {e}"

    # ── PDF 載入 ──────────────────────────────────────────
    def load_pdf(self, pdf_path: str) -> tuple[bool, str]:
        try:
            self.embedding_model = load_embedding_model()
            reader = PdfReader(pdf_path)

            pages_text: list[str] = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

            if not pages_text:
                return False, "❌ PDF 中未提取到任何文字（可能是掃描檔）"

            full_text = "\n".join(pages_text)
            self.chunks = self._split_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
            if not self.chunks:
                return False, "❌ 文本分割後無有效片段"

            raw_embeddings = self.embedding_model.encode(
                self.chunks, convert_to_numpy=True, show_progress_bar=False
            )
            self.embeddings = normalize(raw_embeddings, norm="l2").astype("float32")

            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings)

            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=3000, ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)

            self.pdf_loaded = True
            msg = (
                f"✅ 成功載入！共 {len(reader.pages)} 頁，"
                f"提取 {len(pages_text)} 頁文字，分割為 {len(self.chunks)} 個片段"
            )
            return True, msg

        except Exception as e:
            logger.exception("PDF 載入失敗")
            return False, f"❌ 載入失敗: {e}"

    # ── 文本分割 ──────────────────────────────────────────
    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        paragraphs = re.split(r"\n{2,}", text)
        merged = []
        buffer = ""

        for para in paragraphs:
            para = re.sub(r"\s+", " ", para).strip()
            if not para:
                continue
            if len(buffer) + len(para) + 1 <= chunk_size:
                buffer = f"{buffer} {para}".strip()
            else:
                if buffer:
                    merged.append(buffer)
                while len(para) > chunk_size:
                    cut = para[:chunk_size].rfind("。")
                    if cut == -1:
                        cut = para[:chunk_size].rfind(". ")
                    if cut == -1:
                        cut = chunk_size
                    else:
                        cut += 1
                    merged.append(para[:cut].strip())
                    para = para[max(0, cut - overlap):].strip()
                buffer = para

        if buffer:
            merged.append(buffer)

        return [c for c in merged if len(c) > 20]

    # ── LLM 呼叫 ──────────────────────────────────────────
    def _llm_call(self, prompt: str, system: str = "", max_tokens: int = 200,
                  temperature: float = 0.3) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    # ── 向量搜尋 ──────────────────────────────────────────
    def _vector_search(self, query_text: str, top_k: int,
                       index: faiss.IndexFlatIP = None,
                       chunks: list[str] = None) -> list[tuple[int, float, str]]:
        index = index if index is not None else self.index
        chunks = chunks if chunks is not None else self.chunks

        query_vec = self.embedding_model.encode([query_text], convert_to_numpy=True)
        query_vec = normalize(query_vec, norm="l2").astype("float32")
        scores, indices = index.search(query_vec, min(top_k, len(chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(chunks):
                results.append((int(idx), float(score), chunks[idx]))
        return results

    # ==================== 8 種 RAG 策略 ====================
    def strategy_1_basic_similarity(self, query: str, top_k: int = 3) -> list[str]:
        results = self._vector_search(query, top_k)
        return [chunk for _, _, chunk in results]

    def strategy_2_tfidf(self, query: str, top_k: int = 3) -> list[str]:
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chunks[idx] for idx in top_indices if similarities[idx] > 0]

    def strategy_3_hybrid(self, query: str, top_k: int = 3) -> list[str]:
        k_rrf = 60
        sem_results = self._vector_search(query, top_k=min(top_k * 3, len(self.chunks)))
        sem_ranking = {idx: rank for rank, (idx, _, _) in enumerate(sem_results)}

        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
        tfidf_ranking = {
            idx: rank
            for rank, idx in enumerate(tfidf_scores.argsort()[::-1][:top_k * 3])
        }

        all_indices = set(sem_ranking.keys()) | set(tfidf_ranking.keys())
        rrf_scores = {}
        for idx in all_indices:
            score = 0.0
            if idx in sem_ranking:
                score += 1.0 / (k_rrf + sem_ranking[idx])
            if idx in tfidf_ranking:
                score += 1.0 / (k_rrf + tfidf_ranking[idx])
            rrf_scores[idx] = score

        sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [self.chunks[idx] for idx in sorted_indices[:top_k]]

    def strategy_4_reranking(self, query: str, top_k: int = 3) -> list[str]:
        candidates = self.strategy_1_basic_similarity(query, top_k=top_k * 2)

        prompt = (
            f"問題：{query}\n\n"
            "請對以下文本片段按照與問題的相關度排序（最相關的排前面），"
            "只回傳編號，用逗號分隔，例如：2,1,3\n\n"
        )
        for i, chunk in enumerate(candidates, 1):
            prompt += f"[{i}] {chunk[:200]}…\n\n"

        try:
            result = self._llm_call(prompt, max_tokens=50, temperature=0)
            numbers = [int(n) - 1 for n in re.findall(r"\d+", result)]
            reranked, seen = [], set()
            for n in numbers:
                if 0 <= n < len(candidates) and n not in seen:
                    reranked.append(candidates[n])
                    seen.add(n)
            for i, c in enumerate(candidates):
                if i not in seen:
                    reranked.append(c)
            return reranked[:top_k]
        except Exception:
            logger.warning("重新排序失敗，回退至基礎搜尋")
            return candidates[:top_k]

    def strategy_5_multi_query(self, query: str, top_k: int = 3) -> list[str]:
        expansion_prompt = (
            f"請將以下問題改寫成 3 個不同角度的問題，每行一個，不要編號：\n{query}"
        )
        try:
            result = self._llm_call(expansion_prompt, max_tokens=200, temperature=0.7)
            extra_queries = [
                q.strip().lstrip("0123456789.、-）) ")
                for q in result.split("\n") if q.strip()
            ][:3]
            queries = [query] + extra_queries
        except Exception:
            queries = [query]

        chunk_scores: dict[int, float] = {}
        k_rrf = 60
        for q in queries:
            results = self._vector_search(q, top_k=top_k)
            for rank, (idx, _, _) in enumerate(results):
                chunk_scores[idx] = chunk_scores.get(idx, 0) + 1.0 / (k_rrf + rank)

        sorted_indices = sorted(chunk_scores, key=chunk_scores.get, reverse=True)
        return [self.chunks[idx] for idx in sorted_indices[:top_k]]

    def strategy_6_contextual_compression(self, query: str, top_k: int = 3) -> list[str]:
        chunks = self.strategy_1_basic_similarity(query, top_k=top_k)
        compressed = []
        for chunk in chunks:
            try:
                result = self._llm_call(
                    f"從以下文本中，提取與問題「{query}」最直接相關的 1-3 句話。"
                    f"只輸出提取結果，不要加任何說明：\n\n{chunk}",
                    max_tokens=200, temperature=0,
                )
                compressed.append(result if result else chunk[:300])
            except Exception:
                compressed.append(chunk[:300])
        return compressed

    def strategy_7_parent_child(self, query: str, top_k: int = 3) -> list[str]:
        full_text = " ".join(self.chunks)
        small_chunks = self._split_text(full_text, SMALL_CHUNK_SIZE, SMALL_CHUNK_OVERLAP)

        if not small_chunks:
            return self.strategy_1_basic_similarity(query, top_k)

        small_embeddings = self.embedding_model.encode(small_chunks, convert_to_numpy=True)
        small_embeddings = normalize(small_embeddings, norm="l2").astype("float32")

        small_index = faiss.IndexFlatIP(small_embeddings.shape[1])
        small_index.add(small_embeddings)

        results = self._vector_search(query, top_k=top_k * 2,
                                      index=small_index, chunks=small_chunks)

        parent_chunks, seen = [], set()
        for _, _, small_chunk in results:
            for i, big_chunk in enumerate(self.chunks):
                if i not in seen and small_chunk[:50] in big_chunk:
                    parent_chunks.append(big_chunk)
                    seen.add(i)
                    break
            if len(parent_chunks) >= top_k:
                break

        return parent_chunks if parent_chunks else self.strategy_1_basic_similarity(query, top_k)

    def strategy_8_hypothetical_answer(self, query: str, top_k: int = 3) -> list[str]:
        try:
            hypothetical = self._llm_call(
                f"請針對以下問題，寫一段可能出現在文件中的回答段落（約 100 字）：\n{query}",
                max_tokens=200, temperature=0.7,
            )
        except Exception:
            hypothetical = query

        results = self._vector_search(hypothetical, top_k)
        return [chunk for _, _, chunk in results]

    # ── 主流程 ────────────────────────────────────────────
    STRATEGY_MAP = {
        "1. 基礎語意搜尋": "strategy_1_basic_similarity",
        "2. TF-IDF 關鍵詞": "strategy_2_tfidf",
        "3. 混合搜尋 (RRF)": "strategy_3_hybrid",
        "4. 重新排序": "strategy_4_reranking",
        "5. 多查詢擴展": "strategy_5_multi_query",
        "6. 上下文壓縮": "strategy_6_contextual_compression",
        "7. 父子文檔": "strategy_7_parent_child",
        "8. 假設性答案 (HyDE)": "strategy_8_hypothetical_answer",
    }

    def generate_answer(self, query: str, strategy: str, top_k: int = 3):
        if not self.client:
            return None, None, "❌ 請先設定 API Key！"
        if not self.pdf_loaded:
            return None, None, "❌ 請先上傳 PDF 檔案！"
        if not query.strip():
            return None, None, "⚠️ 請輸入問題"

        try:
            method_name = self.STRATEGY_MAP.get(strategy, "strategy_1_basic_similarity")
            retrieval_func = getattr(self, method_name)
            relevant_chunks = retrieval_func(query, int(top_k))

            if not relevant_chunks:
                return None, None, "⚠️ 未檢索到相關片段，請嘗試其他策略或調整 Top-K"

            context = "\n\n---\n\n".join(relevant_chunks)
            answer = self._llm_call(
                prompt=(
                    f"請根據以下上下文回答問題。如果上下文中沒有相關資訊，請明確說明。\n\n"
                    f"上下文：\n{context}\n\n問題：{query}\n\n請用繁體中文詳細回答："
                ),
                system="你是專業的文件分析助手，回答時引用上下文中的具體內容。",
                max_tokens=1024,
                temperature=0.3,
            )
            return answer, relevant_chunks, None

        except Exception as e:
            logger.exception("生成答案失敗")
            return None, None, f"❌ 生成答案失敗: {e}"


# ══════════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════════
def init_session_state():
    if "rag" not in st.session_state:
        st.session_state.rag = MultiStrategyRAG()
    if "api_key_ok" not in st.session_state:
        st.session_state.api_key_ok = False
    if "pdf_ok" not in st.session_state:
        st.session_state.pdf_ok = False
    if "history" not in st.session_state:
        st.session_state.history = []


def render_sidebar():
    rag: MultiStrategyRAG = st.session_state.rag

    with st.sidebar:
        st.header("⚙️ 系統設定")

        # ── 1. API Key ──
        st.subheader("🔑 步驟 1: Groq API Key")
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
        )
        if st.button("驗證 API Key", use_container_width=True):
            if api_key:
                with st.spinner("驗證中…"):
                    ok, msg = rag.set_api_key(api_key)
                st.session_state.api_key_ok = ok
                (st.success if ok else st.error)(msg)
            else:
                st.warning("請先輸入 API Key")

        if st.session_state.api_key_ok:
            st.caption("🟢 API Key 已驗證")
        else:
            st.caption("🔴 尚未驗證 API Key")

        st.divider()

        # ── 2. PDF 上傳 ──
        st.subheader("📤 步驟 2: 上傳 PDF")
        uploaded_pdf = st.file_uploader(
            "選擇 PDF 檔案",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if st.button("🚀 載入文件", use_container_width=True, type="primary"):
            if uploaded_pdf is None:
                st.warning("請先選擇 PDF 檔案")
            elif not st.session_state.api_key_ok:
                st.warning("請先驗證 API Key")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.read())
                    tmp_path = tmp.name
                with st.status("處理中…", expanded=True) as status:
                    st.write("📖 解析 PDF…")
                    st.write("✂️ 分割文本…")
                    st.write("🧠 生成嵌入向量…")
                    ok, msg = rag.load_pdf(tmp_path)
                    status.update(
                        label=msg,
                        state="complete" if ok else "error",
                        expanded=False,
                    )
                st.session_state.pdf_ok = ok
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if st.session_state.pdf_ok:
            st.caption(f"🟢 已載入 {len(rag.chunks)} 個片段")
        else:
            st.caption("🔴 尚未載入 PDF")

        st.divider()

        # ── 3. 策略設定 ──
        st.subheader("🎯 步驟 3: 策略設定")
        strategy = st.selectbox(
            "RAG 策略",
            options=list(MultiStrategyRAG.STRATEGY_MAP.keys()),
            index=0,
        )
        top_k = st.slider("檢索片段數量 (Top-K)", 1, 10, 3)

        st.divider()

        if st.button("🗑️ 清除對話歷史", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    return strategy, top_k


def render_strategy_table():
    with st.expander("📖 8 種 RAG 策略說明", expanded=False):
        st.markdown(
            """
| # | 策略 | 特點 | 適用場景 |
|---|------|------|---------|
| 1 | 基礎語意搜尋 | 快速、通用 | 一般性問答 |
| 2 | TF-IDF 關鍵詞 | 精確匹配專有名詞 | 查找具體術語 |
| 3 | 混合搜尋 (RRF) | 兼顧語意與關鍵詞 | 平衡型查詢 |
| 4 | 重新排序 | LLM 精排，品質最高 | 對精度要求高 |
| 5 | 多查詢擴展 | 覆蓋面廣 | 模糊問題 |
| 6 | 上下文壓縮 | 精簡雜訊 | 長文件處理 |
| 7 | 父子文檔 | 精準定位 + 完整上下文 | 結構化文檔 |
| 8 | HyDE | 假設性答案嵌入 | 探索性問題 |
            """
        )


def render_main(strategy: str, top_k: int):
    st.title("🤖 多策略 RAG PDF 問答系統")
    st.caption("採用 **8 種不同的 RAG 策略**，為您的 PDF 文件提供智能問答服務")

    render_strategy_table()

    # 範例問題
    st.markdown("##### 💡 範例問題")
    example_cols = st.columns(4)
    examples = [
        "這份文件的主要內容是什麼？",
        "文件中提到哪些重要概念？",
        "有哪些關鍵數據或統計資料？",
        "文件的結論是什麼？",
    ]
    for col, ex in zip(example_cols, examples):
        if col.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending_query = ex

    # 對話歷史
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(item["query"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            with st.expander(
                f"📚 檢索來源（策略：{item['strategy']}，{len(item['chunks'])} 片段）"
            ):
                for i, chunk in enumerate(item["chunks"], 1):
                    st.markdown(f"**片段 {i}**")
                    st.text(chunk)
                    st.divider()

    # 輸入框
    pending = st.session_state.pop("pending_query", None)
    query = st.chat_input("輸入您的問題…")
    if pending and not query:
        query = pending

    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner(f"使用「{strategy}」檢索並生成答案中…"):
                rag: MultiStrategyRAG = st.session_state.rag
                answer, chunks, error = rag.generate_answer(query, strategy, top_k)

            if error:
                st.error(error)
            else:
                st.markdown(answer)
                with st.expander(
                    f"📚 檢索來源（策略：{strategy}，{len(chunks)} 片段）"
                ):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**片段 {i}**")
                        st.text(chunk)
                        st.divider()

                st.session_state.history.append({
                    "query": query,
                    "answer": answer,
                    "chunks": chunks,
                    "strategy": strategy,
                })


def main():
    st.set_page_config(
        page_title="多策略 RAG PDF 問答",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    strategy, top_k = render_sidebar()
    render_main(strategy, top_k)


if __name__ == "__main__":
    main()
