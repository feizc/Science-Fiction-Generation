# Create negative label for petraining
# We select the next sentence as correct,  pervious and future sentence as incorrect label.

import os
import numpy as np 

data_path = os.path.join(os.getcwd(), 'data')
txt_path = os.path.join(data_path, 'txt')
vec_path = os.path.join(data_path, 'vec')
vec_files = os.listdir(vec_path)

# Concatenate all the hidden states in one matrix
vectors = np.zeros((1, 768))

for vec_file in vec_files:
    vec_file_path = os.path.join(vec_path, vec_file)
    mat = np.loadtxt(vec_file_path)
    mat = mat.reshape(-1, 768)
    vectors = np.concatenate((vectors, mat), axis=0)

vectors = vectors[1:]

# Create correct and uncorrect data 
input_vec = vectors[:-1]
correct_label_vec = vectors[1:]
uncorrect_future = np.concatenate((vectors[2:], vectors[0].reshape(-1, 768)), axis=0) 
uncorrect_previous = np.concatenate((vectors[-1].reshape(-1,768), vectors[:-1]), axis=0)

assert len(input_vec) == len(correct_label_vec)

# Save the results. 
np.savetxt(os.path.join(data_path, 'input.txt'), input_vec, fmt='%f')
np.savetxt(os.path.join(data_path, 'correct.txt'), correct_label_vec, fmt='%f')
np.savetxt(os.path.join(data_path, 'future.txt'), uncorrect_future, fmt='%f')
np.savetxt(os.path.join(data_path, 'previous.txt'), uncorrect_previous, fmt='%f')

