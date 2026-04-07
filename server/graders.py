import hashlib
import os
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

class Grader:
    pass

class AccuracyGrader(Grader):
    def score(self, prediction, label):
        if prediction is None or label is None:
            return 0.0
        return 1.0 if prediction == label else 0.0

class FormatGrader(Grader):
    def score(self, action):
        if not action or not hasattr(action, "action_type") or not hasattr(action, "payload"):
            return 0.0
        if action.action_type not in ["READ", "SELECT_POLICY", "ANALYZE", "FINAL_ANSWER", "REVISE"]:
            return 0.0
        if action.action_type == "FINAL_ANSWER":
            if action.payload.prediction is None or action.payload.confidence is None:
                return 0.0
        return 1.0

class PolicySelectionGrader(Grader):
    def score(self, selected_policies, true_policies):
        if not true_policies:
            return {"f1_score": 1.0 if not selected_policies else 0.0, "precision": 0.0, "recall": 0.0, "missed_count": 0}
            
        selected_set = set(selected_policies)
        true_set = set(true_policies)
        
        correct_predicted = len(selected_set.intersection(true_set))
        total_predicted = len(selected_set)
        total_true = len(true_set)
        
        precision = correct_predicted / total_predicted if total_predicted > 0 else 0.0
        recall = correct_predicted / total_true if total_true > 0 else 0.0
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
            
        missed_policies = true_set - selected_set
        return {
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall,
            "missed_count": len(missed_policies)
        }

class EfficiencyGrader(Grader):
    def score(self, steps_used, max_steps):
        return max(0.0, 1.0 - (steps_used / max_steps))

class LengthGrader(Grader):
    """
    R_len: Targets a specific reasoning length to prevent 'length collapse' and 'reward hacking'.
    Based on the paper, we target 100-300 words for detailed reasoning.
    """
    def score(self, reasoning_notes):
        reasoning = " ".join(reasoning_notes)
        word_count = len(reasoning.split())
        
        # R_len: Minimal reasoning check (Prevent length collapse)
        if word_count < 30:
            return 0.0
            
        # Optimal range: 100 - 300 words
        if 100 <= word_count <= 300:
            return 1.0
        elif word_count < 100:
            return word_count / 100.0 # Partial reward for shorter reasoning
        else:
            return max(0.0, 1.0 - (word_count - 300) / 300.0) # Penalize excessive verbosity
        
class LLMReasoningGrader(Grader):
    """
    Original LLM-as-a-Judge reasoning grader.
    Planned to use it for the as the reasoning grader, but was unsure of the 
    judge's env so replaced with a ProgrammaticReasoningGrader
    """
    def __init__(self):
        self.cache = {}
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "dummy")
        
        if os.environ.get("HF_TOKEN"):
            default_base_url = "https://router.huggingface.co/v1"
            default_model = "Qwen/Qwen2.5-72B-Instruct"
        else:
            default_base_url = "https://router.huggingface.co/v1"
            default_model = "gpt-4o-mini"
            
        base_url = os.environ.get("API_BASE_URL") or default_base_url
        self.model_name = os.environ.get("MODEL_NAME") or default_model
        
        if api_key != "dummy":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            except ImportError:
                self.client = None
        else:
            self.client = None

    def score(self, reference_reasoning, reasoning_notes):
        reasoning = " ".join(reasoning_notes)
        if not reasoning:
            return 0.0
            
        cache_key = hashlib.md5((content + reasoning).encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.client:
            return 0.5 # fallback
            
        # R_rub: Rubric-based reward assessing Faithfulness and Factuality.
        # This replaces the simple logic check with a more robust safety/policy audit.
        prompt = f"""Evaluate the content moderation reasoning based on the following rubric:
1. Faithfulness (0.0-0.5): Did the model follow the task instructions and logic steps properly?
2. Factuality (0.0-0.5): Is the reasoning grounded in the provided policy and content facts? 
   Does it avoid 'factuality hallucinations' or post-hoc rationalizations?

Content/Reference: {reference_reasoning}
Reasoning: {reasoning}

Return only a single scalar float score from 0.0 to 1.0 representing the aggregate quality."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            score_str = response.choices[0].message.content.strip()
            match = re.search(r'0\.\d+|1\.0|0|1', score_str)
            score = float(match.group(0)) if match else 0.0
        except Exception:
            score = 0.0
            
        score = max(0.0, min(1.0, score))
        self.cache[cache_key] = score
        return score

class ProgrammaticReasoningGrader(Grader):
    """
    programmatic grader for reasoning evaluation.
    Measures linguistic structure, causal linkage, and semantic relevance.
    """
    def __init__(self):
        self.causal_markers = [
            "because", "therefore", "since", "due to", "consequently", 
            "specifically", "however", "as a result", "leads to", 
            "violates", "consistent with", "implies"
        ]
        # HashingVectorizer: Stateless, fast, and stable for small corpora
        self.vectorizer = HashingVectorizer(n_features=2**18)
        
        for resource in ['punkt_tab', 'punkt', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
            except (LookupError, AttributeError):
                nltk.download(resource, quiet=True)

        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()

        self.generic_patterns = [
            "the content in question",
            "this violates",
            "given these factors",
            "it is clear that",
            "this content"
        ]

    def _score_linguistics(self, reasoning):
        """Analyze structural quality and causal linkage."""
        if not reasoning: return 0.0
        
        sentences = sent_tokenize(reasoning)
        words = [w.lower() for w in word_tokenize(reasoning) if w.isalnum()]
        
        if not words: return 0.0
        
        # 1. Causal Linkage (30%)
        causal_count = sum(1 for m in self.causal_markers if m in reasoning.lower())
        causal_score = min(1.0, causal_count / 3.0)
        
        # 2. Lexical Diversity (TTR) (30%)
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        diversity_score = min(1.0, ttr * 2.0) # Boost TTR for typical reasoning lengths
        
        # 3. Structural Complexity (40%)
        # Reward having multiple sentences (at least 3 for full score)
        sentence_score = min(1.0, len(sentences) / 3.0)
        
        return (0.3 * causal_score) + (0.3 * diversity_score) + (0.4 * sentence_score)

    def _score_relevance(self, reasoning, reference_reasoning):
        """Measure semantic overlap with the source content."""
        if not reasoning or not reference_reasoning: return 0.0
        
        try:
            # HashingVectorizer doesn't need fit, eliminating noise in small corpora
            vecs = self.vectorizer.transform([reference_reasoning, reasoning])
            similarity = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
            
            # Hallucination Penalty: Reasoning must have basic grounding
            if similarity < 0.15:
                return 0.0
                
            # Stricter multiplier (2.5 instead of 5.0) to prevent generic answers from maxing the score
            return min(1.0, similarity * 2.5) 
        except Exception:
            # Fallback to simple keyword overlap
            r_words = set(reasoning.lower().split())
            c_words = set(reference_reasoning.lower().split())
            overlap = len(r_words.intersection(c_words))
            return min(1.0, overlap / 5.0)

    def _policy_grounding_score(self, reasoning, true_policies):
        """Measures explicit policy mentions."""
        if not true_policies:
            return 1.0
        
        reasoning_lower = reasoning.lower()
        matches = sum(1 for p in true_policies if p in reasoning_lower)
        
        return matches / len(true_policies)

    def _generic_penalty(self, reasoning):
        """Penalizes template-based or boilerplate reasoning."""
        reasoning_lower = reasoning.lower()
        hits = sum(1 for g in self.generic_patterns if g in reasoning_lower)
        return min(0.3, hits * 0.1)

    def _context_usage_score(self, reasoning, context_posts):
        """Rewards utilizing provided contextual conversation posts."""
        if not context_posts:
            return 1.0
        
        context_text = " ".join(context_posts).lower()
        reasoning_lower = reasoning.lower()
        
        # Simple word overlap with context (ignoring common words handled by length grader implicitly)
        overlap = sum(1 for word in context_text.split() if len(word) > 3 and word in reasoning_lower)
        
        return min(1.0, overlap / 5.0)

    def _contradiction_penalty(self, reasoning, prediction):
        """Detects contradictions between reasoning sentiment and final labels."""
        reasoning_lower = reasoning.lower()
        
        # Binary prediction (0 = keep, 1 = remove)
        if "not violate" in reasoning_lower and prediction == 1:
            return 0.3
        if "violates" in reasoning_lower and prediction == 0:
            return 0.3
            
        return 0.0

    def score(self, reference_reasoning, reasoning_notes, true_policies=None, context_posts=None, prediction=None):
        reasoning = " ".join(reasoning_notes)
        raw_words = word_tokenize(reasoning)
        unique_words = set(w.lower() for w in raw_words if w.isalnum())
        
        # Stricter minimum requirements for "production ready" reasoning
        if not reasoning or len(raw_words) < 15 or len(unique_words) < 8:
            return 0.0
            
        linguistic_score = self._score_linguistics(reasoning)
        relevance_score = self._score_relevance(reasoning, reference_reasoning)
        policy_score = self._policy_grounding_score(reasoning, true_policies)
        context_score = self._context_usage_score(reasoning, context_posts)
        
        generic_penalty = self._generic_penalty(reasoning)
        contradiction_penalty = self._contradiction_penalty(reasoning, prediction)
        
        # Word Repetition Penalty: Penalize "gaming" by repeating keywords
        word_counts = Counter(w.lower() for w in raw_words if w.isalnum())
        if word_counts and max(word_counts.values()) > len(raw_words) * 0.2:
            linguistic_score *= 0.5
            
        # Penalize extreme repetition (Low TTR)
        ttr = len(unique_words) / len(raw_words)
        if ttr < 0.4:
            linguistic_score *= 0.5
        
        # Final upgraded reasoning formula
        final_score = (
            (0.30 * linguistic_score) + 
            (0.40 * relevance_score) + 
            (0.20 * policy_score) + 
            (0.10 * context_score)
        ) - generic_penalty - contradiction_penalty
        
        return max(0.0, min(1.0, final_score))

# Set the active grader
ReasoningGrader = ProgrammaticReasoningGrader
