#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Q1

class MaxHeap:
  def __init__(self):
    self.heap = []

  # O(log n) time complexity
  def insert(self, value):
    self.heap.append(value)
    self._sift_up(len(self.heap) - 1)

  # O(log n) time complexity
  def delete(self):
    if self.is_empty():
      return None
    self._swap(0, -1)
    removed = self.heap.pop()
    self._sift_down(0)
    return removed

  # O(1) time complexity
  def get_max(self):
    if self.is_empty():
      return None
    return self.heap[0]

  def is_empty(self):
    return len(self.heap) == 0

  # Helper function to move a new element up the heap
  def _sift_up(self, index):
    while index > 0:
      parent_index = (index - 1) // 2
      if self.heap[index] > self.heap[parent_index]:
        self._swap(index, parent_index)
        index = parent_index
      else:
        break

  # Helper function to move an element down the heap (after removal)
  def _sift_down(self, index):
    while (2 * index + 1) < len(self.heap):
      child_index = 2 * index + 1
      # Find the larger child (left or right)
      if child_index + 1 < len(self.heap) and self.heap[child_index + 1] > self.heap[child_index]:
        child_index += 1
      if self.heap[index] >= self.heap[child_index]:
        break
      self._swap(index, child_index)
      index = child_index

  # Helper function to swap elements in the heap
  def _swap(self, index1, index2):
    self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]


# In[2]:


#Q2

import requests

def download_urls_with_retries(urls, max_retries=3):
  """
  Downloads content from a list of URLs with retries on failures.

  Args:
      urls: A list of URLs to download content from.
      max_retries: The maximum number of retries for failed downloads (default 3).

  Returns:
      A dictionary where keys are URLs and values are downloaded content (as strings) 
      or None if download failed after retries.
  """
  results = {}
  for url in urls:
    content = None
    for attempt in range(max_retries + 1):
      try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for non-2xx status codes
        content = response.text
        break  # Download successful, exit loop
      except (requests.exceptions.RequestException, Exception) as e:
        print(f"Error downloading {url} (attempt {attempt+1}/{max_retries}): {e}")
    results[url] = content
  return results


# In[4]:


#Q3

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Boston house price dataset
boston = load_boston()

# Feature matrix (independent variables)
X = boston.data

# Target variable (dependent variable)
y = boston.target

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model coefficients and performance metrics
print("Coefficients:", model.coef_)
print("Mean squared error:", mse)
print("R-squared:", r2)


# In[5]:


#Q4

def clean_and_preprocess(df):
  """
  Clean and preprocess a DataFrame.

  Args:
      df: A pandas DataFrame to be cleaned and preprocessed.

  Returns:
      A new pandas DataFrame with cleaned and preprocessed data.
  """
  
  # Handle missing values (replace with mean or most frequent for numerical/categorical)
  df.fillna(df.mean(numeric_only=True), inplace=True)  # Replace numerical with mean
  for col in df.select_dtypes(include=['object']):
    df[col].fillna(df[col].mode()[0], inplace=True)  # Replace categorical with mode

  # Normalize numerical columns (using StandardScaler)
  numerical_cols = df.select_dtypes(include=[np.number])
  scaler = StandardScaler()
  df[numerical_cols] = scaler.fit_transform(numerical_cols)

  # Encode categorical columns (using OneHotEncoder)
  categorical_cols = df.select_dtypes(include=['object'])
  encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
  encoded_cols = encoder.fit_transform(categorical_cols)
  encoded_cols_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

  # Combine processed numerical and encoded categorical columns
  df_processed = pd.concat([df[numerical_cols], encoded_cols_df], axis=1)

  return df_processed


# In[6]:


#Q5

def fibonacci_recursive(n):
  """
  Calculates the nth Fibonacci number using recursion.

  Args:
      n: The index of the Fibonacci number to calculate (0-based indexing).

  Returns:
      The nth Fibonacci number.

  Raises:
      ValueError: If n is negative.
  """
  if n < 0:
    raise ValueError("n must be non-negative")
  if n == 0 or n == 1:
    return n
  else:
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)


# In[7]:


#Q6

def safe_divide(numerator, denominator):
  """
  Divides two numbers and handles division by zero with a custom error message.

  Args:
      numerator: The number to be divided (dividend).
      denominator: The number to divide by (divisor).

  Returns:
      The result of the division or a custom error message for division by zero.

  Raises:
      ValueError: If the denominator is zero.
  """
  if denominator == 0:
    raise ValueError("Division by zero is not allowed.")
  return numerator / denominator


# In[8]:


#Q7


import time
from functools import wraps

def log_execution_time(logger=None):
  """
  A decorator that measures the execution time of a function and logs it.

  Args:
      logger: An optional logger object with a method called "info" to log messages.
              If not provided, will print to the console.

  Returns:
      A decorator function.
  """
  
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      start_time = time.perf_counter()
      result = func(*args, **kwargs)
      end_time = time.perf_counter()
      elapsed_time = end_time - start_time
      log_message = f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to execute."
      if logger:
        logger.info(log_message)
      else:
        print(log_message)
      return result
    return wrapper
  return decorator

# Define a computationally expensive function (example: large prime number calculation)
def find_large_prime(n):
  """
  Finds a large prime number greater than n (simple but slow implementation).
  """
  num = n + 1
  while True:
    if is_prime(num):
      return num
    num += 1

def is_prime(num):
  if num <= 1:
    return False
  for i in range(2, int(num**0.5) + 1):
    if num % i == 0:
      return False
  return True

# Apply the decorator to the expensive function
@log_execution_time()
def find_large_prime_decorated(n):
  return find_large_prime(n)

# Example usage
large_prime = find_large_prime_decorated(100000)
print(f"Found large prime: {large_prime}")


# In[12]:


#Q8

def perform_calculation(num1, operator, num2):
  """
  Performs an arithmetic operation based on the provided operator.

  Args:
      num1: The first number.
      operator: The operator string (+, -, *, /).
      num2: The second number.

  Returns:
      The result of the operation or None if the operator is invalid.

  Raises:
      ZeroDivisionError: If the operator is division and the divisor is zero.
  """
  if operator == "+":
    return num1 + num2
  elif operator == "-":
    return num1 - num2
  elif operator == "*":
    return num1 * num2
  elif operator == "/":
    if num2 == 0:
      raise ZeroDivisionError("Division by zero is not allowed.")
    return num1 / num2
  else:
    return None  # Invalid operator

# Example usage
result = perform_calculation(5, "+", 3)
print(result)  # Output: 8

result = perform_calculation(10, "-", 2)
print(result)  # Output: 8

result = perform_calculation(4, "*", 6)
print(result)  # Output: 24

try:
  result = perform_calculation(12, "/", 0)
except ZeroDivisionError as e:
  print(e)  # Output: Division by zero is not allowed.


# In[10]:


#Q9

import random
import string

def generate_random_password(length=12):
  """
  Generates a random password with a mix of uppercase letters, lowercase letters, 
  digits, and special characters.

  Args:
      length: The desired length of the password (default 12).

  Returns:
      A random password string.
  """

  # Define character sets for different types
  uppercase_letters = string.ascii_uppercase
  lowercase_letters = string.ascii_lowercase
  digits = string.digits
  special_characters = string.punctuation

  # Combine all character sets
  all_characters = uppercase_letters + lowercase_letters + digits + special_characters

  # Ensure at least one character from each set is present
  password = [random.choice(uppercase_letters)]
  password.append(random.choice(lowercase_letters))
  password.append(random.choice(digits))
  password.append(random.choice(special_characters))

  # Fill the remaining slots with random characters from the combined set
  password.extend(random.choices(all_characters, k=length - 4))

  # Shuffle the password characters for better randomness
  random.shuffle(password)

  # Join the characters into a string
  return ''.join(password)

# Example usage
password = generate_random_password()
print(password)


# In[11]:


#Q10

def transpose_matrix(matrix):
  """
  Transposes a 2D list (matrix).

  Args:
      matrix: A 2D list representing the matrix.

  Returns:
      A new 2D list representing the transposed matrix.
  """
  
  # Handle empty matrix or matrix with single row
  if not matrix or not matrix[0]:
    return []
  
  # Get the number of rows and columns
  rows = len(matrix)
  cols = len(matrix[0])

  # Create an empty list to store the transposed matrix
  transposed = [[None] * rows for _ in range(cols)]

  # Fill the transposed matrix
  for i in range(rows):
    for j in range(cols):
      transposed[j][i] = matrix[i][j]

  return transposed

# Example usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed_matrix = transpose_matrix(matrix)

print("Original matrix:")
for row in matrix:
  print(row)

print("\nTransposed matrix:")
for row in transposed_matrix:
  print(row)


# In[ ]:




