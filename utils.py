import math
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

# Define the CNN model
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

# Define the GRU model
class BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_prob=0.2):
        super(BiGRU, self).__init__()
        
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=drop_prob, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        output = self.fc(gru_out)
        return output

# Function to extract features using CNN
def extract_features(cnn_model, data):
    cnn_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        features = cnn_model(data.permute(0, 2, 1).to(device))
        features = features.permute(0, 2, 1)
    return features

# Function to create DataLoader
def create_dataloader(X, y, batch_size):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def evaluateDataset_1DCNN(dataset, cnn_model, rnn_model):
    sensors = ['s_2','s_3','s_4','s_7','s_8','s_9','s_11','s_12','s_13','s_14','s_15','s_17','s_20','s_21']
    window_size = 30
    alpha = 0.1
    threshold = 125
    sensor_size = len(sensors)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(dataset, sensors, window_size, alpha, threshold)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert data to tensors and move to device
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Ensure the model is in evaluation mode and move it to the correct device
    cnn_model = CNNFeatureExtractor(input_channels=sensor_size).to(device)
    cnn_model.eval()
    rnn_model.eval()
    cnn_model.to(device)
    rnn_model.to(device)

    # Extract features using CNN for test set
    test_X_features = extract_features(cnn_model, X_test_tensor)

    # Use the model to predict y_test
    with torch.no_grad():
        predictions_tensor = rnn_model(test_X_features)
        
        # Clamp the predictions to be within [0, 125]
        predictions_tensor = torch.clamp(predictions_tensor, 0, 125)
        
        predictions = predictions_tensor.cpu().numpy()

    # Convert y_test_tensor to numpy array
    y_actual_test = y_test_tensor.cpu().numpy()

    def evaluate(y_true, y_hat, label='test'):
        mse = mean_squared_error(y_true, y_hat)
        rmse = np.sqrt(mse)
        variance = r2_score(y_true, y_hat)
        
        score = 0
        for i in range(len(y_hat)):
            if y_true[i] <= y_hat[i]:
                score = score + np.exp(-(y_true[i] - y_hat[i]) / 10.0) - 1
            else:
                score = score + np.exp((y_true[i] - y_hat[i]) / 13.0) - 1
        
        print('{} RMSE {} R2 {} score {}'.format(dataset, rmse, variance, score))

    # Evaluate the model's performance on the test set
    evaluate(y_actual_test, predictions, label='test')

def evaluateDataset(dataset, model_pytorch):
    # sensors to work with: 14 sensors
    #sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
    sensors = ['s_2','s_3','s_4','s_7','s_8','s_9','s_11','s_12','s_13','s_14','s_15','s_17','s_20','s_21']
    window_size = 30
    alpha = 0.1
    threshold = 125
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(dataset, sensors, window_size, alpha, threshold)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Ensure the model is in evaluation mode and move it to the correct device
    model_pytorch.eval()
    model_pytorch.to(device)

    # Use the model to predict y_test
    with torch.no_grad():
        #model.load_state_dict(torch.load('C:/Users/User/exploring-nasas-turbofan-dataset/weights/linear_model_weights_20_5.pth'))
        predictions_tensor = model_pytorch(X_test_tensor)
        
        # Clamp the predictions to be within [0, 125]
        predictions_tensor = torch.clamp(predictions_tensor, 0, 125)
        
        predictions = predictions_tensor.cpu().numpy()

    # Convert y_test_tensor to numpy array
    y_actual_test = y_test_tensor.cpu().numpy()

    def evaluate(y_true, y_hat, label='test'):
        mse = mean_squared_error(y_true, y_hat)
        rmse = np.sqrt(mse)
        variance = r2_score(y_true, y_hat)
        
        score = 0
        #y_true = y_true.cpu()
        #y_hat = y_hat.cpu()
        for i in range(len(y_hat)):
            if y_true[i] <= y_hat[i]:
                score = score + np.exp(-(y_true[i] - y_hat[i]) / 10.0) - 1
            else:
                score = score + np.exp((y_true[i] - y_hat[i]) / 13.0) - 1
        
        print('{} RMSE {} R2 {} score {}'.format(dataset, rmse, variance, score))

    # Evaluate the model's performance on the test set
    evaluate(y_actual_test, predictions, label='test')

def mixUp(X1, y1, X2, y2, lambda_beta):
    assert len(X1) == len(y1) and len(X2) == len(y2), "Data and labels must have the same length"
    
    # 샘플링할 데이터의 인덱스 결정
    min_len = min(len(X1), len(X2))
    indices = np.random.permutation(len(X1))[:min_len]
    indices2 = np.random.permutation(len(X2))[:min_len]
    lambda_vals = np.random.beta(lambda_beta, lambda_beta, size=len(indices))
    
    # MixStyle 적용
    X_mixed = lambda_vals[:, None, None] * X1[indices] + (1 - lambda_vals[:, None, None]) * X2[indices2]
    y_mixed = lambda_vals[:, None] * y1[indices] + (1 - lambda_vals[:, None]) * y2[indices2]
    
    return X_mixed, y_mixed


# --------------------------------------- DATA PRE-PROCESSING ---------------------------------------
def complementaryAttack(df, percentage):
    data_copy = df.reshape(-1)
    # 125가 아닌 값들에 대한 불리언 마스크 생성
    mask = (data_copy != 125)
    # 125가 아닌 값들만 필터링
    y_train_filtered = data_copy[mask]
    sample_indices = np.random.choice(len(y_train_filtered), size=int(len(y_train_filtered)*percentage), replace=False)
    sampled_data = y_train_filtered[sample_indices].reshape(-1) 
    
    # KDE로 PDF 추정
    kde = gaussian_kde(data_copy)
    x_grid = np.linspace(min(data_copy), max(data_copy), 1000)
    pdf = kde.evaluate(x_grid)

    # CDF 계산
    cdf = cumtrapz(pdf, x_grid, initial=0)
    cdf /= cdf[-1]  # CDF를 정규화

    # 주어진 x 값에 대한 CDF 값을 계산하기 위한 선형 보간 함수 생성
    cdf_interp = interp1d(x_grid, cdf, kind='linear')

    offset_list = []

    for i in sampled_data:
        cdf_value_at_x = cdf_interp(i)
        temp_percentage = cdf_value_at_x
        
        offset_percentage = 0

        if temp_percentage >= 0.5:
            offset_percentage = temp_percentage - 0.5
        else:
            offset_percentage = temp_percentage + 0.5

        origin_x = np.interp(temp_percentage, cdf, x_grid)
        offset_x = np.interp(offset_percentage, cdf, x_grid)
        
        offset_list.append(offset_x)
        
        print(f"origin_x value (CDF={temp_percentage*100:.2f}%): {origin_x}")
        print(f"offset_x value (CDF={offset_percentage*100:.2f}%): {offset_x}")

    offset_list = np.array(offset_list)    
    y_train_filtered[sample_indices] = offset_list
    data_copy[mask] = y_train_filtered
    
    print(offset_list.shape, df.shape, data_copy.shape)
    
    return data_copy.reshape(-1,1)

def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    
    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    
    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame

def add_operating_condition(df):
    df_op_cond = df.copy()
    
    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))
    
    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                        df_op_cond['setting_2'].astype(str) + '_' + \
                        df_op_cond['setting_3'].astype(str)
    
    return df_op_cond

def condition_scaler(df_train, df_test, sensor_names):
    # apply operating condition specific scaling
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_train.loc[df_train['op_cond']==condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_test.loc[df_test['op_cond']==condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond']==condition, sensor_names])
    return df_train, df_test


def exponential_smoothing(df, sensors, n_samples, alpha):
    df = df.copy()
    # first, take the exponential weighted mean
    df[sensors] = df.groupby('unit_nr')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)
    
    # second, drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result
    
    mask = df.groupby('unit_nr')['unit_nr'].transform(create_mask, samples=n_samples).astype(bool)
    df = df[mask]
    
    return df


def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]

    # -1 and +1 because of Python indexing
    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data[start:stop, :]
        
def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()
        
    data_gen = (list(gen_train_data(df[df['unit_nr']==unit_nr], sequence_length, columns))
               for unit_nr in unit_nrs)
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    return data_array

def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # -1 because I want to predict the rul of that last row in the sequence, not the next row
    return data_matrix[sequence_length-1:num_elements, :]  

def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()
        
    label_gen = [gen_labels(df[df['unit_nr']==unit_nr], sequence_length, label) 
                for unit_nr in unit_nrs]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return label_array

def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value) # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:,:] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values
        
    # specifically yield the last possible sequence
    stop = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :]  
    
	
def get_data(dataset, sensors, sequence_length, alpha, threshold):
	# files  
	dir_path = '../CMAPSSData/' 
	train_file = 'train_'+dataset+'.txt' 
	test_file = 'test_'+dataset+'.txt' 
     
     
    # columns 
	index_names = ['unit_nr', 'time_cycles'] 
	setting_names = ['setting_1', 'setting_2', 'setting_3'] 
	sensor_names = ['s_{}'.format(i+1) for i in range(0,21)] 
	col_names = index_names + setting_names + sensor_names 
     
     
    # data readout 
	train = pd.read_csv((dir_path+train_file), sep=r'\s+', header=None, 
					 names=col_names)
	test = pd.read_csv((dir_path+test_file), sep=r'\s+', header=None, 
					 names=col_names)
	y_test = pd.read_csv((dir_path+'RUL_'+dataset+'.txt'), sep=r'\s+', header=None, 
					 names=['RemainingUsefulLife']).clip(upper=threshold)


    # create RUL values according to the piece-wise target function
	train = add_remaining_useful_life(train)
	train['RUL'].clip(upper=threshold, inplace=True)


    # remove unused sensors
	drop_sensors = [element for element in sensor_names if element not in sensors]

    # scale with respect to the operating condition
	X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
	X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
	X_train_pre, X_test_pre = condition_scaler(X_train_pre, X_test_pre, sensors)

    # exponential smoothing
	X_train_pre= exponential_smoothing(X_train_pre, sensors, 0, alpha)
	X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)

	# train-val split
	gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
	# generate the train/val for *each* sample -> for that we iterate over the train and val units we want
	# this is a for that iterates only once and in that iterations at the same time iterates over all the values we want,
	# i.e. train_unit and val_unit are not a single value but a set of training/vali units
	for train_unit, val_unit in gss.split(X_train_pre['unit_nr'].unique(), groups=X_train_pre['unit_nr'].unique()): 
		train_unit = X_train_pre['unit_nr'].unique()[train_unit]  # gss returns indexes and index starts at 1
		val_unit = X_train_pre['unit_nr'].unique()[val_unit]

		x_train = gen_data_wrapper(X_train_pre, sequence_length, sensors, train_unit)
		y_train = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], train_unit)
		
		x_val = gen_data_wrapper(X_train_pre, sequence_length, sensors, val_unit)
		y_val = gen_label_wrapper(X_train_pre, sequence_length, ['RUL'], val_unit)

	# create sequences for test 
	test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit_nr']==unit_nr], sequence_length, sensors, -99.))
			   for unit_nr in X_test_pre['unit_nr'].unique())
	x_test = np.concatenate(list(test_gen)).astype(np.float32)
	
	return x_train, y_train, x_val, y_val, x_test, y_test['RemainingUsefulLife']
# ---------------------------------------------------------------------------------------------------
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.model = model
        self.val_loss_min = val_loss
        
