import torch

def maximum_likelihood_estimation_of_Dirichlet_distribution(x: torch.Tensor, iter_FPI: int = 10, eps_FPI: float = 1e-8, iter_Newton: int = 10, eps_Newton: float = 1e-8):
    """
    Args:
        x: [n_samples, dim]

    Returns:
        alpha: [dim]
    """
    ## Initialization
    alpha = (x[:,0].mean() - (x[:,0]**2).mean()) \
            / ((x[:,0]**2).mean() - x[:,0].mean()**2)
    alpha_new = alpha * x.mean(axis=0) # [dim]
    alpha_old = 1e+6
    
    ## Fixed Point Iteration
    count_FPI = 0
    while(count_FPI < iter_FPI and torch.linalg.norm(alpha_new - alpha_old) > eps_FPI):
        alpha_old = alpha_new.copy()
        objective = torch.special.psi(alpha_old.sum()) + torch.log(x).mean(axis=0) #[dim]
        
        count_Newton = 0
        alpha_new = torch.where(objective>=-2.22, 
                                torch.exp(objective)+0.5,
                                -1/(objective - torch.special.psi(1))) #[dim]
        alpha_before = 1e+6
        while(count_Newton < iter_Newton and torch.linalg.norm(alpha_new - alpha_before) > eps_Newton):
            alpha_before = alpha_new.copy()
            alpha_new = alpha_before - (torch.special.psi(alpha_before) - objective) / torch.special.polygamma(1, alpha_before)
            count_Newton += 1
        
        count_FPI += 1
    
    return alpha_new


def compute_dirichlet_log_density(x: torch.Tensor, alpha: torch.Tensor):    
    """
    Args:
        x: [n_samples, dim]
        alpha: [dim]

    Returns:
        log_density: [n_samples]
    """
    lnB = torch.special.gammaln(alpha).sum() - torch.special.gammaln(alpha.sum())
    kernel = torch.sum((torch.special.xlogy(alpha - 1, x.T)).T, 0)
    return -lnB + kernel


def compute_anomalous_scores_by_dirichlet(x: torch.Tensor, y: torch.Tensor):
    """
    Args:
        x: [n_samples, dim]
        y: [n_samples, dim]

    Returns:
        anomalous_scores: [n_samples]
    """
    alpha = maximum_likelihood_estimation_of_Dirichlet_distribution(x)
    log_density = compute_dirichlet_log_density(y, alpha)
    return - log_density