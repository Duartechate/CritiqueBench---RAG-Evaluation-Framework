from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from config import Config
import json
import os

class RAGEvaluator:
    def __init__(self):
        self.eval_llm = ChatOpenAI(
            model_name=Config.EVALUATION_MODEL,
            temperature=0
        )
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
    def evaluate_response(self, question: str, response: str, context: str) -> Dict:
        """Evaluate a RAG response using multiple metrics"""
        metrics = {}
        
        # LLM-based evaluation
        metrics.update(self._llm_evaluation(question, response, context))
        
        
        
        return metrics
    
    def _llm_evaluation(self, question: str, response: str, context: str) -> Dict:
        """Use LLM-as-judge to evaluate response quality"""
        evaluations = {}
        
        for metric in Config.METRICS:
            prompt = self._create_eval_prompt(metric, question, response, context)
            chain = prompt | self.eval_llm | StrOutputParser()
            result = chain.invoke({})
            
            try:
                score = float(result)
                evaluations[f"{metric}_score"] = score
                evaluations[f"{metric}_explanation"] = self._get_explanation(metric, score)
            except ValueError:
                evaluations[f"{metric}_score"] = 0.0
                evaluations[f"{metric}_explanation"] = "Evaluation failed"
        
        return evaluations
    
    def _create_eval_prompt(self, metric: str, question: str, response: str, context: str) -> ChatPromptTemplate:
        """ evaluation prompt for specific metric"""
        templates = {
            "faithfulness": """Evaluate the faithfulness of the response to the context on a scale from 0 to 1.
            Score 1 if all information in the response can be verified in the context.
            Score 0 if the response contains information not present in or contradictory to the context.
            
            Question: {question}
            Context: {context}
            Response: {response}
            
            Provide only the numerical score between 0 and 1.""",
            
            "answer_relevance": """Evaluate how relevant the response is to the question on a scale from 0 to 1.
            Score 1 if the response completely answers the question.
            Score 0 if the response is irrelevant to the question.
            
            Question: {question}
            Response: {response}
            
            Provide only the numerical score between 0 and 1.""",
            
            "context_relevance": """Evaluate how relevant the context is to the question on a scale from 0 to 1.
            Score 1 if all parts of the context are relevant to answering the question.
            Score 0 if none of the context is relevant to the question.
            
            Question: {question}
            Context: {context}
            
            Provide only the numerical score between 0 and 1."""
        }
        
        return ChatPromptTemplate.from_template(
            templates[metric]
        ).partial(
            question=question,
            response=response,
            context=context
        )
    
    def _traditional_metrics(self, response: str, references: List[str]) -> Dict:
        """Calculate traditional NLP metrics"""
        metrics = {}
        
        # BLEU
        tokenized_response = word_tokenize(response.lower())
        tokenized_refs = [word_tokenize(ref.lower()) for ref in references]
        metrics["bleu_score"] = sentence_bleu(tokenized_refs, tokenized_response)
        
        # ROUGE (using first reference)
        scores = self.scorer.score(references[0], response)
        for key in scores:
            metrics[f"{key}_f1"] = scores[key].fmeasure
        
        return metrics
    
    def _get_explanation(self, metric: str, score: float) -> str:
        """Generate explanation for the score"""
        explanations = {
            "faithfulness": [
                "The response contains significant hallucinations or contradictions",
                "The response has some minor inaccuracies",
                "The response is mostly faithful to the context",
                "The response is completely faithful to the context"
            ],
            "answer_relevance": [
                "The response is completely irrelevant to the question",
                "The response partially addresses the question",
                "The response mostly answers the question",
                "The response completely answers the question"
            ],
            "context_relevance": [
                "The context is completely irrelevant to the question",
                "Some parts of the context are relevant",
                "Most of the context is relevant",
                "All context is highly relevant"
            ]
        }
        
        index = min(3, int(score * 4))
        return explanations[metric][index]
    
    def save_results(self, results: Dict, filename: str):
        """Save evaluation results to JSON file"""
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(Config.RESULTS_DIR, filename)
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)