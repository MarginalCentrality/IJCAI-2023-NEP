import numpy as np


def get_avg_imp(init_robustness, imp_sols):
    """
    Args:
        init_robustness:
        imp_sols: [... [robustness, sol] ...]

    Returns: Average robustness improvement
    """
    final_robust = np.array([_[0] for _ in imp_sols])
    return np.mean(final_robust - init_robustness)


def get_max_imp(init_robustness, imp_sols):
    """
    Args:
        init_robustness:
        imp_sols: [... [robustness, sol] ...]

    Returns: max robustness improvement
    """
    final_robust = np.array([_[0] for _ in imp_sols])
    return np.max(final_robust - init_robustness)


def get_n_diff_optimal_sols(imp_sols):
    """
    Args:
        imp_sols: [... [robustness, sol] ...]

    Returns: Number of different optimal solutions.
    """

    def get_normalized_repr(sol):
        """

        Args:
            sol: [u_{1}, u_{1'} ,..., u_{2m}, u_{2m'}]

        Returns:
        """
        assert len(sol) % 2 == 0, 'Length of the solution is not an even number~'
        edge_pairs = []
        for i in range(0, len(sol), 2):
            u, v = sol[i], sol[i + 1]
            edge_pair = (u, v) if u < v else (v, u)
            edge_pairs.append(edge_pair)

        edge_pairs.sort()

        return '-'.join(['-'.join([str(edge_pair[0]), str(edge_pair[1])]) for edge_pair in edge_pairs])

    largest_robustness = max([imp_sol[0] for imp_sol in imp_sols])

    best_sol = set()
    for robustness, sol in imp_sols:
        if robustness == largest_robustness:
            best_sol.add(get_normalized_repr(sol))

    return len(best_sol)



