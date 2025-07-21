from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, pipeline

class RAGAnswerValidator:
    def __init__(self, client, rag_model_name="facebook/rag-token-nq"):
        self.tokenizer = RagTokenizer.from_pretrained(rag_model_name)
        self.retriever = RagRetriever.from_pretrained(rag_model_name, index_name="exact")
        self.rag_model = RagSequenceForGeneration.from_pretrained(rag_model_name)
        self.qa_pipeline = pipeline("question-answering")
        self.client = client

    def check_answer(self, question, answer, cells):
        """
        Validates if the answer is derived from the cells using RAG and GPT.
        """
        cell_content = self._get_cell_content(cells)

        rag_answer = self._get_rag_answer(question, cell_content)

        if rag_answer:
            is_valid = self.client.validate_answer(question, answer, rag_answer)
            return is_valid

        return False

    def _get_cell_content(self, cells):
        """Concatenates cell content into a single context string."""
        content = ""
        for cell_number, cell in cells:
            if cell.cell_type == 'markdown':
                content += cell.source + "\n"
            elif cell.cell_type == 'code':
                content += cell.source + "\n"
                for output in cell.outputs:
                    if hasattr(output, 'text'):
                        content += output.text + "\n"
        return content

    def _get_rag_answer(self, question, cell_content):
        """Uses RAG to generate an answer based on cell content."""
        try:
            inputs = self.tokenizer(question, return_tensors="pt")
            self.retriever.index_references([cell_content])
            outputs = self.rag_model.generate(
                input_ids=inputs["input_ids"],
                num_beams=2,
                num_return_sequences=1
            )
            rag_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return rag_answer
        except Exception as e:
            print(f"Error in RAG generation: {e}")
            return None
