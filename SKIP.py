import numpy as np

def compute_cumulative_query(q, m):
    """
    Compute the cumulative sum and cumulative sum of squares for query q.
    Corresponds to pseudocode line 1.
    Time Complexity: O(m)
    
    Parameters:
      q: numpy array representing the query time series (length m)
      m: int, length of the query
      
    Returns:
      S_1_q: cumulative sum array of q
      S_1_q2: cumulative sum array of q^2
    """
    S_1_q = np.cumsum(q)
    S_1_q2 = np.cumsum(q**2)
    return S_1_q, S_1_q2

def compute_alpha_cumulative_arrays(O, q, m, α):
    """
    Compute the α-skipping cumulative arrays for each object in O.
    This corresponds to pseudocode line 2.
    Each object o has three arrays:
      S_α_o: cumulative sum of o at α intervals
      S_α_o2: cumulative sum of o^2 at α intervals
      S_α_qo: cumulative sum of q * o at α intervals
    Time Complexity per object: O(m), overall: O(n*m)
    
    Parameters:
      O: list or 2D array of time series objects (each of length m)
      q: query time series (numpy array)
      m: int, length of the series
      α: int, α-skipping interval
      
    Returns:
      S_α_o, S_α_o2, S_α_qo: numpy arrays of shape (num_objects, m)
    """
    num_objects = len(O)
    S_α_o = np.zeros((num_objects, m))
    S_α_o2 = np.zeros((num_objects, m))
    S_α_qo = np.zeros((num_objects, m))
    
    for o_idx, o in enumerate(O):
        for i in range(m):
            if i % α == 0:
                if i == 0:
                    S_α_o[o_idx, i] = o[i]
                    S_α_o2[o_idx, i] = o[i] ** 2
                    S_α_qo[o_idx, i] = q[i] * o[i]
                else:
                    # The cumulative sum is computed in O(α) time per update.
                    S_α_o[o_idx, i] = S_α_o[o_idx, i - α] + sum(o[i - α + 1 : i + 1])
                    S_α_o2[o_idx, i] = S_α_o2[o_idx, i - α] + sum(o[i - α + 1 : i + 1] ** 2)
                    S_α_qo[o_idx, i] = S_α_qo[o_idx, i - α] + sum(q[i - α + 1 : i + 1] * o[i - α + 1 : i + 1])
    return S_α_o, S_α_o2, S_α_qo

def compute_initial_sums(o, q, l, S_alpha_o_row, S_alpha_o2_row, S_alpha_qo_row, α):
    """
    Compute the initial running sums for object o when τ = 0.
    This corresponds to pseudocode lines 5-7 for τ = 0.
    It uses the precomputed α-skipping cumulative arrays.
    Time Complexity: O(α)
    
    Parameters:
      o: numpy array representing the object time series
      q: query time series (numpy array)
      l: int, the length of the subsequence
      S_alpha_o_row, S_alpha_o2_row, S_alpha_qo_row: precomputed α-skipping cumulative arrays for object o
      α: int, the α-skipping interval
      
    Returns:
      sum_o: cumulative sum of o[0:l]
      sum_o2: cumulative sum of o^2[0:l]
      sum_qo: cumulative sum of q*o over [0:l]
    """
    nearest_valid_index = (l - 1) - ((l - 1) % α)
    sum_o  = S_alpha_o_row[nearest_valid_index]
    sum_o2 = S_alpha_o2_row[nearest_valid_index]
    sum_qo = S_alpha_qo_row[nearest_valid_index]
    for i in range(nearest_valid_index + 1, l):
        sum_o  += o[i]
        sum_o2 += o[i] ** 2
        sum_qo += q[i] * o[i]
    return sum_o, sum_o2, sum_qo

def update_incrementally_sums(o, q, τ, l, curr_sum_o, curr_sum_o2, curr_sum_qo):
    """
    Update the running sums for object o when τ > 0.
    This corresponds to pseudocode lines 5-7 for the else-case.
    It subtracts the value at index τ-1 and adds the value at index τ+l-1.
    Time Complexity: O(1)
    
    Parameters:
      o: numpy array representing the object time series
      q: query time series (numpy array)
      τ: int, current start index of the subsequence
      l: int, length of the subsequence
      curr_sum_o, curr_sum_o2, curr_sum_qo: current running sums
      
    Returns:
      Updated running sums: curr_sum_o, curr_sum_o2, curr_sum_qo
    """
    curr_sum_o  = curr_sum_o - o[τ - 1] + o[τ + l - 1]
    curr_sum_o2 = curr_sum_o2 - o[τ - 1]**2 + o[τ + l - 1]**2
    curr_sum_qo = curr_sum_qo - q[τ - 1] * o[τ - 1] + q[τ + l - 1] * o[τ + l - 1]
    return curr_sum_o, curr_sum_o2, curr_sum_qo

def update_incrementally_query_sums(q, curr_sum_q, curr_sum_q2, τ, l):
    """
    Update the running cumulative sums for the query q when τ > 0.
    Corresponds to the query part in pseudocode lines 5-7.
    Time Complexity: O(1)
    
    Parameters:
      q: query time series (numpy array)
      curr_sum_q, curr_sum_q2: current running sums for q and q^2
      τ: int, current start index of the subsequence
      l: int, length of the subsequence
      
    Returns:
      Updated running sums: curr_sum_q, curr_sum_q2
    """
    curr_sum_q  = curr_sum_q - q[τ - 1] + q[τ + l - 1]
    curr_sum_q2 = curr_sum_q2 - q[τ - 1]**2 + q[τ + l - 1]**2
    return curr_sum_q, curr_sum_q2

def compute_pearson_correlation_from_sums(sum_q, sum_q2, sum_o, sum_o2, sum_qo, l):
    """
    Compute the Pearson correlation coefficient (ρ) from the given running sums.
    This function calculates the mean and standard deviation for both q and o,
    and then computes ρ according to:
    
      ρ = (sum_qo - l * μ_q * μ_o) / (l * σ_q * σ_o)
    
    Corresponds to pseudocode line 12.
    Time Complexity: O(1)
    
    Parameters:
      sum_q, sum_q2: cumulative sum and sum of squares for query subsequence
      sum_o, sum_o2: cumulative sum and sum of squares for object subsequence
      sum_qo: cumulative sum of element-wise products (q * o)
      l: int, length of the subsequence
      
    Returns:
      ρ: Pearson correlation coefficient for the subsequence pair.
    """
    μ_q = sum_q / l
    σ_q = np.sqrt(sum_q2 / l - μ_q**2)
    
    μ_o = sum_o / l
    σ_o = np.sqrt(sum_o2 / l - μ_o**2)
    
    ρ = (sum_qo - l * μ_q * μ_o) / (l * σ_q * σ_o)
    return ρ

def is_overlapping(new_candidate, existing_candidates):
    """
    Check if the new candidate subsequence overlaps with any existing candidate
    from the same object.
    
    Two subsequences are considered overlapping if their time intervals intersect.
    
    Parameters:
      new_candidate: tuple (object index, start index, length, ρ)
      existing_candidates: list of candidate tuples
      
    Returns:
      True if overlapping with any candidate from the same object, otherwise False.
    """
    new_o, new_start, new_length, _ = new_candidate
    new_end = new_start + new_length - 1
    for candidate in existing_candidates:
        o, start, length, _ = candidate
        if o == new_o:
            end = start + length - 1
            if not (new_end < start or end < new_start):
                return True
    return False

def skip_algorithm(q, O, m, α, δ, k):
    """
    SKIP Algorithm for finding the most correlated subsequences.
    
    This function searches through the database O for subsequences
    whose Pearson correlation with query q exceeds the threshold δ.
    
    The algorithm processes in a top-down manner by decreasing subsequence length l.
    For each (τ, l) combination, the cumulative sums for q and each object are
    computed/updated. The Pearson correlation is then computed in O(1) time using
    precomputed cumulative sums. The use of α-skipping cumulative arrays reduces
    the per-correlation computation from O(l) to O(α).
    
    Corresponding pseudocode lines:
      - Line 1: Compute cumulative sums for q.
      - Line 2: Compute α-skipping cumulative arrays for each object.
      - Lines 5-7: Compute/update running sums for each subsequence.
      - Line 12: Compute the Pearson correlation.
    
    Parameters:
      q: numpy array representing the query time series (length m)
      O: list/2D-array of time series (each of length m)
      m: int, length of the query
      α: int, α-skipping interval
      δ: float, correlation coefficient threshold
      k: int, maximum number of subsequences to retrieve
      
    Returns:
      R: list of tuples (object index, start index, subsequence length, ρ)
         for the top k subsequences that meet the correlation threshold.
    """
    num_objects = len(O)
    R = []  # List to store resulting candidate subsequences
    
    # --- Pseudocode line 1: Compute cumulative sums for query q ---
    S_1_q, S_1_q2 = compute_cumulative_query(q, m)
    
    # --- Pseudocode line 2: Compute α-skipping cumulative arrays for each object ---
    S_α_o, S_α_o2, S_α_qo = compute_alpha_cumulative_arrays(O, q, m, α)
    
    # Initialize arrays to hold running sums for each object (for each (τ, l))
    sum_o_arr  = np.zeros(num_objects)
    sum_o2_arr = np.zeros(num_objects)
    sum_qo_arr = np.zeros(num_objects)
    
    # Initialize variables for query running sums (will be updated per τ)
    sum_q_current  = None
    sum_q2_current = None
    
    # Loop over subsequence lengths from m down to 1 (pseudocode line 3)
    for l in range(m, 0, -1):
        # Loop over start index τ from 0 to m - l (pseudocode line 4)
        for τ in range(0, m - l + 1):
            # --- Query subsequence cumulative sums ---
            if τ == 0:
                # For τ = 0, directly use precomputed cumulative sums (O(1))
                sum_q_current  = S_1_q[l - 1]
                sum_q2_current = S_1_q2[l - 1]
            else:
                # For τ > 0, update running sums incrementally in O(1)
                sum_q_current, sum_q2_current = update_incrementally_query_sums(q, sum_q_current, sum_q2_current, τ, l)
            
            # Process each object in the database
            for o_idx, o in enumerate(O):
                if τ == 0:
                    # For τ = 0: compute initial running sums using α-skipping arrays (O(α))
                    (sum_o_arr[o_idx],
                     sum_o2_arr[o_idx],
                     sum_qo_arr[o_idx]) = compute_initial_sums(o, q, l,
                                                               S_α_o[o_idx],
                                                               S_α_o2[o_idx],
                                                               S_α_qo[o_idx],
                                                               α)
                else:
                    # For τ > 0: update running sums incrementally (O(1))
                    (sum_o_arr[o_idx],
                     sum_o2_arr[o_idx],
                     sum_qo_arr[o_idx]) = update_incrementally_sums(o, q, τ, l,
                                                                     sum_o_arr[o_idx],
                                                                     sum_o2_arr[o_idx],
                                                                     sum_qo_arr[o_idx])
                
                # --- Pseudocode line 12: Compute Pearson correlation ---
                ρ = compute_pearson_correlation_from_sums(sum_q_current, sum_q2_current,
                                                          sum_o_arr[o_idx], sum_o2_arr[o_idx],
                                                          sum_qo_arr[o_idx], l)
                
                # If the correlation exceeds threshold δ, and the candidate does not overlap
                # with any existing candidate from the same object, add it to the result.
                new_candidate = (o_idx, τ, l, ρ)
                if ρ > δ and not is_overlapping(new_candidate, R):
                    R.append(new_candidate)
                    if len(R) == k:
                        return R
    return R
