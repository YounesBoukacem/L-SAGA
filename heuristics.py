import numpy as np



#=====================================================================================
                                #Palmer
#=====================================================================================
class Palmer:
    def __init__(self, jobs_list:list):
        self.jobs_list = jobs_list
        self.nb_jobs = len(jobs_list)
        self.nb_machines = len(jobs_list[0])
        self.seq_star = None
        self.make_span_star = None

    # utility function that returns the gantt cumule based on a job execution times and a previous gantt cumule
    def cumulate(self, job:list, previous_cumul=None):
        res = [0] * len(job)

        if(previous_cumul==None):
            res[0] = job[0]
            for i in range(1, len(job)):
                res[i] = res[i-1] + job[i]
        else:
            res[0] = previous_cumul[0] + job[0]
            for i in range(1, len(job)):
                res[i] = max(res[i-1], previous_cumul[i]) + job[i]

        return res

    # utility function that computes the gantt cumule given only a job sequence (not used in the algorithm due to inneficiency
    # dynamic programming with cumulate is used instead ...)
    def cumulate_seq(self, seq:list):
        cumulated = None
        for i in seq:
            cumulated = self.cumulate(self. jobs_list[i], cumulated)

        return cumulated


    #launching the optimization
    def optim(self, debug=False):
        jobs_weights=[]
        for i, job in zip(range(self.nb_jobs), self.jobs_list):
            weight = 0
            for j in range(self.nb_machines):
                if(debug==True):
                    print(f">job {i} mach {j} first term: {(2*(j+1) - 1) - self.nb_machines}")
                    print(f">job {i} mach {j} second term: {job[j]}")
                    print("------------------------------------------------------------------")
                weight += ((2*(j+1) - 1)- self.nb_machines) * job[j]
            if(debug==True):
                print(f"===>> job {i} weight: {weight}")
            jobs_weights.append((weight, i))
        
        self.seq_star = [tu[1] for tu in sorted(jobs_weights, reverse=True)]
        self.make_span_star = self.cumulate_seq(self.seq_star)[-1]
        
        return (self.seq_star, self.make_span_star)


class Inst:
    def __init__(self, jobs: int, machines: int, seed: int, ub: int, lb: int, matrix: list[list[int]]):
        self.jobs = jobs
        self.machines = machines
        self.seed = seed
        self.ub = ub
        self.lb = lb
        self.matrix = matrix

    def __repr__(self) -> str:
        return f'Inst(jobs={self.jobs}, machines={self.machines}, seed={self.seed}, ub={self.ub}, lb={self.lb}, matrix={self.matrix})'

#=====================================================================================
                                #NEH
#=====================================================================================

class NEH():
    def __init__(self, instance: Inst, debug: bool = False):
        self.instance = instance
        self.debug = debug

    def calculate_sj(self, job: int) -> int:
        sj = 0
        for machine in range(self.instance.machines):
            sj += self.instance.matrix[machine][job]
        return sj

    def sort_jobs(self, reverse: bool = False) -> list[int]:
        return sorted(range(self.instance.jobs), key=lambda job: self.calculate_sj(job), reverse=reverse)

    def emulate(self, jobs: list[int]) -> list[int]:
        machines_exec = [0] * self.instance.machines
        for job in jobs:
            for current_machine in range(self.instance.machines):
                # Add jobs execution time to current machine
                machines_exec[current_machine] += self.instance.matrix[current_machine][job]

                # Sync other machines if they are behind current time
                for machine in range(current_machine + 1, self.instance.machines):
                    machines_exec[machine] = max(machines_exec[current_machine], machines_exec[machine])

        return machines_exec

    def calculate_cmax(self, jobs: list[int]) -> int:
        return self.emulate(jobs)[-1]
    
    def get_best_order(self, orders: list[list[int]]) -> tuple[int, list[int]]:
        min_cmax = float('inf')
        min_order = None
        for order in orders:
            cmax = self.calculate_cmax(order)
            if cmax < min_cmax:
                min_cmax = cmax
                min_order = order

        return min_cmax, min_order

    def get_best_position(self, order: list[int], job: int) -> tuple[int, list[int]]:
        possible_orders: list[list[int]] = []
        for pos in range(len(order) + 1):
            possible_orders.append(order[:pos] + [job] + order[pos:])

        return self.get_best_order(possible_orders)

    def __call__(self) -> tuple[int, list[int]]:
        if self.instance.jobs < 2:
            raise ValueError("Number of jobs must be greater than 2")

        sorted_jobs = self.sort_jobs()
        current_cmax, current_order = self.get_best_order([sorted_jobs[:2], sorted_jobs[:2][::-1]])

        if self.debug:
            print(current_cmax, current_order)

        if self.instance.jobs == 2:
            return current_cmax, current_order
        
        for job in sorted_jobs[2:]:
            current_cmax, current_order = self.get_best_position(current_order, job)
            if self.debug:
                print(current_cmax, current_order)
        
        return current_cmax, current_order


# Function to cumulate job processing times
def cumulate(job, previous_cumul=None):
    res = [0] * len(job)
    if previous_cumul is None:
        res[0] = job[0]
        for i in range(1, len(job)):
            res[i] = res[i - 1] + job[i]
    else:
        res[0] = previous_cumul[0] + job[0]
        for i in range(1, len(job)):
            res[i] = max(res[i - 1], previous_cumul[i]) + job[i]
    return res


#=====================================================================================
                                #CDS
#=====================================================================================

# Function to cumulate processing times for a given sequence of jobs
def cumulate_seq(seq, jobs_list):
    cumulated = None
    for i in seq:
        cumulated = cumulate(jobs_list[i], cumulated)
    return cumulated
# Function to compute the makespan given a sequence of jobs and the job list
def makespan(sequence, job_list):
    return cumulate_seq(sequence, job_list)[-1]
# Function to perform the Johnson's algorithm for the flow shop problem
def johnson_algorithm(matrix):
    n = matrix.shape[0]
    sequence = []
    machines = [[], []]

    # Preprocessing to determine the order of jobs
    for i in range(n):
        if matrix[i][0] < matrix[i][1]:  # if time(m1) < time(m2)
            machines[0].append((matrix[i][0], i))
        else:
            machines[1].append((matrix[i][1], i))

    # Sorting jobs for each machine
    machines[0] = sorted(machines[0], key=lambda x: x[0])                    # ascending sort for the first machine
    machines[1] = sorted(machines[1], key=lambda x: x[0], reverse=True)      # descending sort for the second machine

    # Merging the two sorted lists
    merged = machines[0] + machines[1]

    # Constructing the optimal sequence
    sequence = [index for _, index in merged]

    return sequence
# Function that applies Johnson's algorithm and computes the makespan
def johnson(job_matrix, data_matrix):
    sequence = johnson_algorithm(job_matrix)
    return sequence, makespan(sequence, data_matrix)

# CDS heuristic
def cds_heuristic(matrix):
    n = matrix.shape[0]
    m = matrix.shape[1]
    best_makespan = float('inf')
    best_sequences = []

    # Step 1: Generate matrices of all possible job lists
    for i in range(1, m):
        machine_subset_1 = matrix[:, :i].sum(axis=1)
        machine_subset_2 = matrix[:, -i:].sum(axis=1)
        job_matrix = np.column_stack((machine_subset_1, machine_subset_2))
             
        # Step 2: Apply Johnson's algorithm to the job matrix abd calculate the makespan
        sequence, makespan_value = johnson(job_matrix, matrix)

        # Step 3: Update the best makespan and corresponding sequences
        if makespan_value < best_makespan:
            best_makespan = makespan_value
            best_sequences = [sequence]
        elif makespan_value == best_makespan:
            best_sequences.append(sequence)
            
    return best_sequences[0], best_makespan