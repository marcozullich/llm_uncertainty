import transformers
import torch

from scipy.special import digamma

def get_token_confidence(generation_with_logits: transformers.generation.utils.GenerateDecoderOnlyOutput, apply_softmax:bool=True, top_k:int=1):
    logits = torch.cat(generation_with_logits.scores)
    probs = torch.nn.functional.softmax(logits, dim=-1) if apply_softmax else logits
    per_token_confidence = probs.topk(k=top_k, dim=-1).values
    return per_token_confidence

def token_uncertainty_naive(generation_with_logits: transformers.generation.utils.GenerateDecoderOnlyOutput):
    return 1 - torch.prod(per_token_confidence)

def token_uncertainty_vanilla(generation_with_logits: transformers.generation.utils.GenerateDecoderOnlyOutput):
    return (1 - per_token_confidence).mean()

def logTokU_epistemic(confidences: torch.Tensor):
    return confidences.shape[1] / (confidences + 1).sum(dim=1)

def logTokU_aleatoric(confidences: torch.Tensor):
    alpha_0 = confidences.sum(dim=1)
    aleatoric_confidences = []
    for confidences_row, alpha_0_row in zip(confidences, alpha_0):
        digamma_deltas = digamma(confidences_row + 1) - digamma(alpha_0_row + 1)
        digamma_deltas_sum = ((confidences_row / alpha_0_row) * digamma_deltas).sum()
        aleatoric_confidences.append(-digamma_deltas_sum)
    return torch.Tensor(aleatoric_confidences)

def logTokU(generation_with_logits: transformers.generation.utils.GenerateDecoderOnlyOutput, top_k_inconfident:int = 5):
    '''
    Implements Logits-Induced Token Uncertainty as in the paper
    https://arxiv.org/abs/2412.14737
    '''

    per_token_confidence = get_token_confidence(generation_with_logits, apply_softmax=False, top_k=top_k_inconfident)
    epistemic_uncertainty = logTokU_epistemic(per_token_confidence)
    aleatoric_uncertainty = logTokU_aleatoric(per_token_confidence)
    
    reliability_scores = - epistemic_uncertainty * aleatoric_uncertainty
    reliability_total = 1/top_k_inconfident * reliability_scores.topk(k=top_k_inconfident).values.sum()

    return reliability_total
    
    
