import json
import re
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Import từ config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CORPUS_FILE, CHUNK_FILE, DATA_DIR

class CorpusBuilder:
    def __init__(self):
        self.corpus_file = CORPUS_FILE
        self.chunk_file = CHUNK_FILE
        self.graph_file = DATA_DIR / "graphs" / "citation_edges.json"
        
    def _split_sentences(self, text):
        """Tách câu dựa trên dấu chấm hoặc xuống dòng"""
        sentences = re.split(r'(?<=\.)\s+|\n', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _split_paragraphs_with_overlap(self, text):
        """Tách đoạn với 50% sliding window overlap"""
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 20]
        chunks = []
        for i in range(len(paragraphs)):
            chunk = paragraphs[i]
            if i < len(paragraphs) - 1:
                chunk += " " + paragraphs[i+1]
            chunks.append(chunk)
        return chunks

    def build_chunks(self, data):
        """Tạo Multi-Granularity Chunks (tự động dò key JSON)"""
        all_chunks = []
        chunk_id_counter = 0
        
        # Nếu data là dict (ALQAC hay dùng dạng này), chuyển thành list
        law_list = data if isinstance(data, list) else data.values()
        
        for law in tqdm(law_list, desc="Building Chunks"):
            # Thử nhiều key khác nhau để lấy law_id
            law_id = str(law.get("law_id", law.get("id", "")))
            
            for article in law.get("articles", []):
                # Thử nhiều key khác nhau để lấy article_id
                article_id = str(article.get("article_id", article.get("id", "")))
                text = article.get("text", "")
                
                # 1. Article Level
                all_chunks.append({"chunk_id": f"chunk_{chunk_id_counter}", "law_id": law_id, "article_id": article_id, "text": text, "level": "article"})
                chunk_id_counter += 1
                
                # 2. Paragraph Level
                for para in self._split_paragraphs_with_overlap(text):
                    all_chunks.append({"chunk_id": f"chunk_{chunk_id_counter}", "law_id": law_id, "article_id": article_id, "text": para, "level": "paragraph"})
                    chunk_id_counter += 1
                    
                # 3. Sentence Level
                for sent in self._split_sentences(text):
                    all_chunks.append({"chunk_id": f"chunk_{chunk_id_counter}", "law_id": law_id, "article_id": article_id, "text": sent, "level": "sentence"})
                    chunk_id_counter += 1
                    
        return all_chunks

    def extract_citation_edges(self, data):
        """Trích xuất liên kết đồ thị thông minh"""
        edges = []
        
        # Pattern 1: Tường minh (Ví dụ: Điều 17 Luật số 12/2015/QH13)
        # Chỉ lấy các mã chứa số, chữ hoa, hoặc ký tự gạch chéo
        pattern_explicit = re.compile(r"Điều\s+(\d+).*?(Luật|Nghị định|Thông tư)\s+([A-Z0-9/_-]+)")
        
        # Pattern 2: Dẫn chiếu nội bộ (Ví dụ: Điều 17 của Luật này)
        pattern_implicit = re.compile(r"Điều\s+(\d+).*?(Luật|Nghị định|Thông tư)\s+này", re.IGNORECASE)
        
        law_list = data if isinstance(data, list) else data.values()
        
        for law in tqdm(law_list, desc="Extracting Graph Edges"):
            source_law = str(law.get("law_id", law.get("id", "")))
            
            for article in law.get("articles", []):
                source_article = str(article.get("article_id", article.get("id", "")))
                text = article.get("text", "")
                
                # Bắt tường minh
                for match in pattern_explicit.findall(text):
                    target_article_id = match[0]
                    target_law_id = f"{match[1]} {match[2]}".strip()
                    edges.append({
                        "source": f"{source_law}_{source_article}",
                        "target": f"{target_law_id}_{target_article_id}",
                        "weight": 1.0
                    })
                
                # Bắt nội bộ ("Luật này") -> target law chính là source law
                for match in pattern_implicit.findall(text):
                    target_article_id = match[0]
                    edges.append({
                        "source": f"{source_law}_{source_article}",
                        "target": f"{source_law}_{target_article_id}",
                        "weight": 1.0
                    })
                    
        # Lọc bỏ các cạnh rác (nếu regex vẫn lọt) và cạnh tự chỉ trỏ vào chính mình
        clean_edges = []
        for e in edges:
            if "n_" not in e["target"] and e["source"] != "_" and e["source"] != e["target"]:
                clean_edges.append(e)
                
        return clean_edges

    def run(self):
        print(f"Reading corpus from {self.corpus_file}...")
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        chunks = self.build_chunks(data)
        edges = self.extract_citation_edges(data)
        
        print(f"Saving {len(chunks)} multi-granular chunks...")
        with open(self.chunk_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        print(f"Saving {len(edges)} citation edges...")
        with open(self.graph_file, 'w', encoding='utf-8') as f:
            json.dump(edges, f, ensure_ascii=False, indent=2)
            
        print("Data Preparation Completed Successfully! 🚀")