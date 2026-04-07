import sys
import os

# Add the current directory to sys.path so we can import server.graders
sys.path.append(os.getcwd())

from server.graders import ProgrammaticReasoningGrader

def test_grader():
    grader = ProgrammaticReasoningGrader()
    
    # 1. Test Policy Grounding
    true_policies = ["hate_speech"]
    valid_reasoning = ["This post violates the hate_speech policy because of the slurs used."]
    score_with_policy = grader.score("reference", valid_reasoning, true_policies=true_policies)
    
    no_policy_reasoning = ["This post is very bad and should be removed immediately."]
    score_without_policy = grader.score("reference", no_policy_reasoning, true_policies=true_policies)
    
    # 2. Test Context Usage
    context_posts = ["User1: 'I love cats'", "User2: 'Cats are great'"]
    context_reasoning = ["The user is responding to a discussion about cats and their greatness."]
    score_with_context = grader.score("cats cats cats", context_reasoning, context_posts=context_posts)
    
    no_context_reasoning = ["The user is just talking normally about things."]
    score_no_context = grader.score("cats cats cats", no_context_reasoning, context_posts=context_posts)
    
    # 3. Test Contradiction
    contradiction_reasoning = ["This does not violate any rule."]
    score_contradiction = grader.score("reference", contradiction_reasoning, prediction=1) # 1 = violate
    
    # 4. Test Generic Penalty
    generic_reasoning = ["The content in question violates our rules. Given these factors, it is clear that this content should be removed."]
    score_generic = grader.score("reference", generic_reasoning)

    print(f"Policy Grounding Score: {score_with_policy:.2f} (with) vs {score_without_policy:.2f} (without)")
    print(f"Context Usage Score: {score_with_context:.2f} (with) vs {score_no_context:.2f} (without)")
    print(f"Contradiction Score: {score_contradiction:.2f}")
    print(f"Generic Penalty Score: {score_generic:.2f}")

if __name__ == "__main__":
    test_grader()
