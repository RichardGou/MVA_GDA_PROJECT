import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import plotly.graph_objects as go


torch.manual_seed(42)
np.random.seed(42)

# Teacher Network
def generate_teacher(d, m0):
    w1 = torch.randn(m0, d)
    w1 = w1 / w1.norm(dim=1, keepdim=True)  # Normalize rows
    w2 = torch.sign(torch.randn(m0))
    return w1, w2

def teacher_output(X, w1, w2):
    return torch.sum(w2[:, None] * torch.relu(w1 @ X.T), dim=0)*100

# Student Network
class StudentNetwork(nn.Module):
    def __init__(self, m, d, scale =1):
        super(StudentNetwork, self).__init__()
        self.hidden_weights = nn.Parameter(scale * torch.randn(m, d))
        self.output_weights = nn.Parameter(scale * torch.randn(m))
    
    def forward(self, X):
        hidden_output = torch.relu(self.hidden_weights @ X.T)  # (m, n)
        return torch.sum(self.output_weights[:, None] * hidden_output, dim=0)  # (n,)

# Training and Evaluation
def train_student(X_train, Y_train, X_test, Y_test, W_init_scale, stepsize, niter, scaling, m):
    # Initialize the student network
    student = StudentNetwork(m, d, W_init_scale)
    optimizer = optim.SGD(student.parameters(), lr=stepsize / scaling)

    train_losses = []
    test_losses = []
    final_params = []

    for _ in range(niter):
        # Forward pass
        Y_pred_train = scaling * student(X_train)
        train_loss =  0.5 * torch.mean((Y_pred_train - Y_train) ** 2) #(1/scaling**2)* 
        train_losses.append(train_loss.item())

        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Test loss
        with torch.no_grad():
            Y_pred_test = scaling * student(X_test)
            test_loss = 0.5 * torch.mean((Y_pred_test - Y_test) ** 2) #(1/scaling**2)*
            test_losses.append(test_loss.item())
        
        if _ == niter - 1:
            params = torch.cat([student.hidden_weights.data.flatten(), student.output_weights.data.flatten()])
            final_params.append(params.cpu().detach().numpy())

    
    return train_losses, test_losses, np.array(final_params)



def pytorch_gd(X_train, Y_train, X_test, Y_test, W_init_scale, stepsize, niter, scaling, m, W_init):
    #converting W_init to desired shape
    hidden_weights = torch.tensor(W_init[:, :d], dtype=torch.float32, requires_grad=True)  # (m, d)
    output_weights = torch.tensor(W_init[:, d].reshape(m), dtype=torch.float32, requires_grad=True)  # (m,)
    
    train_losses = []
    test_losses = []
    
    n = X_train.size(0)
    
    for _ in range(niter):
        #frwrd pass
        hidden_preact = torch.matmul(hidden_weights, X_train.T)  # (m, n)
        hidden_output_train = torch.max(hidden_preact, torch.zeros_like(hidden_preact))  # ReLU activation

        #grad for RELU
        relu_grad_indicator = (hidden_preact > 0).float()

        output_train = scaling * torch.matmul(output_weights, hidden_output_train)#scaling * torch.sum(output_weights[:, None] * hidden_output_train, dim=0)  # (n,)

        #calc loss
        train_loss = 0.5 * torch.mean((output_train - Y_train) ** 2)
        train_losses.append(train_loss.item())
        
        #backprop:
        gradR = (output_train - Y_train) / n  # (n,)
        
        #grad for hidden weights
        grad_w1 = torch.matmul(gradR.unsqueeze(0) * output_weights[:, None] * relu_grad_indicator, X_train)  # (m, d)
        #grad for output
        grad_w2 = torch.sum(hidden_output_train * gradR, dim=1)  # (m,)
        
        #update
        hidden_weights.data -= (stepsize / scaling) * grad_w1
        output_weights.data -= (stepsize / scaling) * grad_w2
        
        #test loss
        hidden_preact_test = torch.matmul(hidden_weights, X_test.T)  # (m, n_test)
        hidden_output_test = torch.max(hidden_preact_test, torch.zeros_like(hidden_preact_test))  # ReLU activation
        output_test = scaling * torch.sum(output_weights[:, None] * hidden_output_test, dim=0)  # (n_test,)
        
        test_loss = 0.5 * torch.mean((output_test - Y_test) ** 2)
        test_losses.append(test_loss.item())
    

    return (train_losses, test_losses, 
            torch.cat([hidden_weights.flatten(), output_weights.flatten()]).cpu().detach().numpy())





#########################experiments
# Parameters
d = 100  # Dimension of the supervised learning problem
n_train, n_test = 1000, 1000  # Number of data points
m0 = 3  # Number of neurons in the ground truth
niter = 25000#WAS 25000 in the original experiment  # Number of iterations
ms = [128]#[6, 8, 12, 16, 24, 32, 64, 128, 256]#, 512]  # Student network sizes#[1, 2, 3, 4, 6, 8]#, 
ntrials = 1  # Number of trials


# Main Experiment
m_ltrains = np.zeros((niter, len(ms), ntrials))
m_ltests = np.zeros((niter, len(ms), ntrials))
m_ltrains2 = np.zeros((niter, len(ms), ntrials))
m_ltests2 = np.zeros((niter, len(ms), ntrials))

# pbar = tqdm(total=len(ms) * ntrials * 2)

# for k in range(ntrials):
#     # Generate teacher network and datasets
#     w1_teacher, w2_teacher = generate_teacher(d, m0)

#     X_train = torch.randn(n_train, d)
#     X_train = X_train / X_train.norm(dim=1, keepdim=True)  # Normalize to unit sphere
#     Y_train = teacher_output(X_train, w1_teacher, w2_teacher)

#     X_test = torch.randn(n_test, d)
#     X_test = X_test / X_test.norm(dim=1, keepdim=True)
#     Y_test = teacher_output(X_test, w1_teacher, w2_teacher)

#     # Compute with alpha = 1/sqrt(m)
#     for i, m in enumerate(ms):
#         scaling = 1 / np.sqrt(m)
#         stepsize = 1 / m
#         W_init_scale = scaling
#         train_losses, test_losses = train_student(X_train, Y_train, X_test, Y_test, W_init_scale, stepsize, niter, scaling, m)
#         m_ltrains[:, i, k] = train_losses
#         m_ltests[:, i, k] = test_losses
#         pbar.update(1)

#     # Compute with alpha = 1/m
#     for i, m in enumerate(ms):
#         scaling = 1 / m
#         stepsize = 0.05 / m
#         W_init_scale = scaling
#         train_losses, test_losses = train_student(X_train, Y_train, X_test, Y_test, W_init_scale, stepsize, niter, scaling, m)
#         m_ltrains2[:, i, k] = train_losses
#         m_ltests2[:, i, k] = test_losses
#         pbar.update(1)

# pbar.close()



        ###############################
"""
def evaluate_loss_landscape(X_train, Y_train, w_teacher, pca_basis, center_point, grid_size=21, grid_span=5):
    grid_x, grid_y = np.meshgrid(
        np.linspace(-grid_span, grid_span, grid_size),
        np.linspace(-grid_span, grid_span, grid_size)
    )
    
    # Generate grid of parameters in PCA space
    parameter_grid = []
    for i in range(grid_size):
        for j in range(grid_size):
            point = center_point + grid_x[i, j] * pca_basis[0] + grid_y[i, j] * pca_basis[1]
            parameter_grid.append(point)
    
    parameter_grid = np.array(parameter_grid)
    losses = []

    for params in parameter_grid:
        # Map back to parameters and compute loss
        with torch.no_grad():
            student.hidden_weights.data = torch.tensor(params[:m * d].reshape(m, d))
            student.output_weights.data = torch.tensor(params[m * d:])
            Y_pred_train = scaling * student(X_train)
            loss = 0.5 * torch.mean((Y_pred_train - Y_train) ** 2).item()
        losses.append(loss)

    return grid_x, grid_y, np.array(losses).reshape(grid_size, grid_size)


# Experiment and Visualization Loop
final_params_all = []
for k in range(ntrials):
    w1_teacher, w2_teacher = generate_teacher(d, m0)

    X_train = torch.randn(n_train, d)
    X_train = X_train / X_train.norm(dim=1, keepdim=True)
    Y_train = teacher_output(X_train, w1_teacher, w2_teacher)

    for i, m in enumerate(ms):
        scaling = 1 / np.sqrt(m)
        stepsize = 1 / m
        W_init_scale = scaling
        train_losses, test_losses, final_params = train_student(
            X_train, Y_train, X_test, Y_test, W_init_scale, stepsize, niter, scaling, m
        )
        final_params_all.append(final_params)

"""


# Function to perform PCA and reduce parameter space to 2D
def perform_pca(final_params_all):
    # Flatten the final parameters into a matrix (each column corresponds to a different parameter vector)
    all_params = np.vstack(final_params_all)
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_params)
    
    # Return PCA components and the reduced parameter set
    return pca.components_, pca_result


# Generate Loss Landscape in PCA Space
def generate_loss_surface(X_train, Y_train, student, pca_basis, center_point, scaling, grid_size=21, grid_span=5):
    # Generate grid in the PCA space
    grid_x, grid_y = np.meshgrid(
        np.linspace(-grid_span[0], grid_span[1], grid_size),
        np.linspace(-grid_span[0], grid_span[1], grid_size)
    )
    
    pca_grid_x = np.zeros_like(grid_x)
    pca_grid_y = np.zeros_like(grid_y)

    losses = np.zeros((grid_size, grid_size))
    # parameter_grid = []
    for i in range(grid_size):
        for j in range(grid_size):
            point = center_point + grid_x[i, j] * pca_basis[0] + grid_y[i, j] * pca_basis[1]
            with torch.no_grad():
                student.hidden_weights.data = torch.tensor(point[:m * d].reshape(m, d))
                student.output_weights.data = torch.tensor(point[m * d:])
                Y_pred_train = scaling * student(X_train)
                loss =  0.5 * torch.mean((Y_pred_train - Y_train) ** 2).item() #(1/scaling**2) *
                losses[i, j] = loss
                centered = point - center_point
                pca_grid_x[i,j] = np.dot(centered, pca_basis[0])
                pca_grid_y[i,j] = np.dot(centered, pca_basis[1])
            # parameter_grid.append(point)
    
    # parameter_grid = np.array(parameter_grid)
    # losses = []

    # for params in parameter_grid:
    #     # Map the PCA grid back to model parameters
    #     with torch.no_grad():
    #         student.hidden_weights.data = torch.tensor(params[:m * d].reshape(m, d))
    #         student.output_weights.data = torch.tensor(params[m * d:])
    #         Y_pred_train = scaling * student(X_train)
    #         loss = (1/(scaling **2)) * 0.5 * torch.mean((Y_pred_train - Y_train) ** 2).item()
    #     losses.append(loss)

    return pca_grid_x, pca_grid_y, losses#np.array(losses).reshape(grid_size, grid_size)




######################## filter-wise normalization################################
# Function for filter-wise normalization
def normalize_filterwise_direction(weights, direction):
    """
    Normalize the direction for each filter (neuron) by the norm of the corresponding filter (neuron).
    
    :param weights: The current weights of the layer (e.g., hidden or output layer)
    :param direction: A random direction to perturb the weights
    :return: Normalized direction (same shape as the weights)
    """
    # Compute the Frobenius norm of each row (for fully connected layers, each row is a filter)
    norms = weights.norm(dim=1, keepdim=True)  # Shape (m, d) for hidden weights or (m,) for output weights
    normalized_direction = direction / norms
    return normalized_direction


# Modify the loss surface generation by using normalized random directions
def generate_loss_surface_with_filterwise(X_train, Y_train, student, pca_basis, center_point, grid_size=21, grid_span=5):
    # Generate grid in the PCA space
    grid_x, grid_y = np.meshgrid(
        np.linspace(-grid_span, grid_span, grid_size),
        np.linspace(-grid_span, grid_span, grid_size)
    )
    
    parameter_grid = []
    for i in range(grid_size):
        for j in range(grid_size):
            point = center_point + grid_x[i, j] * pca_basis[0] + grid_y[i, j] * pca_basis[1]
            parameter_grid.append(point)
    
    parameter_grid = np.array(parameter_grid)
    losses = []

    for params in parameter_grid:
        # Map the PCA grid back to model parameters
        with torch.no_grad():
            student.hidden_weights.data = torch.tensor(params[:m * d].reshape(m, d))
            student.output_weights.data = torch.tensor(params[m * d:])
            
            # Generate random perturbation direction and apply filter-wise normalization
            direction_hidden = torch.randn_like(student.hidden_weights.data)
            direction_output = torch.randn_like(student.output_weights.data)
            
            # Normalize directions for each layer
            direction_hidden_normalized = normalize_filterwise_direction(student.hidden_weights.data, direction_hidden)
            direction_output_normalized = normalize_filterwise_direction(student.output_weights.data, direction_output)
            
            # Add the normalized perturbation to the weights
            student.hidden_weights.data += direction_hidden_normalized
            student.output_weights.data += direction_output_normalized
            
            # Compute the loss with updated weights
            Y_pred_train = scaling * student(X_train)
            loss = 0.5 * torch.mean((Y_pred_train - Y_train) ** 2).item()
        losses.append(loss)

    return grid_x, grid_y, np.array(losses).reshape(grid_size, grid_size)

###################################################################3


# Plotting the Loss Surface and Final Parameters
def plot_loss_landscape(grid_x, grid_y, loss_values, pca_result, trained_loss, ms):
    # Create a 3D surface plot for the loss landscape
    fig = go.Figure(data=[go.Surface(z=loss_values, x=grid_x, y=grid_y, colorscale='Viridis', opacity=0.9)])
    
    # Plot the final model parameters (final_params_all contains the final parameters of the student model)
    for i, pca_proj in enumerate(pca_result):
        # Use pca_proj for final parameters of each m
        m = ms#[i % len(ms)]  # Ensure to map ms correctly
        fig.add_trace(go.Scatter3d(
            x=[pca_proj[0]], y=[pca_proj[1]], z=[trained_loss[i]], 
            mode='markers', marker=dict(size=5, color='red'), name=f"Final Params m={m}"
        ))

    # Layout of the plot
    # fig.update_layout(
    #     scene=dict(
    #         xaxis_title="PCA 1",
    #         yaxis_title="PCA 2",
    #         zaxis_title="Loss"
    #     ),
    #     title="Loss Landscape for Student Network (PCA Projection)"
    # )

    fig.write_html(f"/home/bibahaduri/MVA_GDA_visuals/MVA_GDA_visuals_sqrt{ms}_again")
    #fig.show()


# Main code for plotting loss landscapes after the experiments
final_params_all = []
for k in range(ntrials):

    w1_teacher, w2_teacher = generate_teacher(d, m0)

    X_train = torch.randn(n_train, d)
    X_train = X_train / X_train.norm(dim=1, keepdim=True) # Normalize to unit sphere
    Y_train = teacher_output(X_train, w1_teacher, w2_teacher)

    X_test = torch.randn(n_test, d)
    X_test = X_test / X_test.norm(dim=1, keepdim=True)
    Y_test = teacher_output(X_test, w1_teacher, w2_teacher)
    print("starting the experiments:")
    trained_test_losses = []
    trained_test_loss_sqrt = []
    for i, m in enumerate(ms):
        scaling = 1 / m#np.sqrt(m)
        scaling_sqrt = 1 / np.sqrt(m)
        stepsize_simple = 0.05/m
        stepsize = 1 / m
        W_init_scale = 1 #scaling
        trained_test_losses = []
        trained_test_loss_sqrt = []
        final_params_all = []
        final_params_all_sqrt = []
        for j in range(4):
            init_weights = torch.randn(m, d + 1)

            train_losses, test_losses, final_params = pytorch_gd(
                X_train, Y_train, X_test, Y_test, W_init_scale, stepsize_simple, niter, scaling, m, init_weights
            )
            final_params_all.append(final_params)  # Store the final parameters of the model
            trained_test_losses.append(test_losses)
        # Perform PCA on final parameters
        trained_test_losses = np.stack(trained_test_losses, axis=0)
        trained_test_losses = np.mean(trained_test_losses, axis=0)
        #pca_basis, pca_result = perform_pca(final_params_all)

        #sqrt scaling
        for j in range(4):
            train_losses, test_losses, final_params = pytorch_gd(
                X_train, Y_train, X_test, Y_test, W_init_scale, stepsize, niter, scaling_sqrt, m, init_weights
            )
            final_params_all_sqrt.append(final_params)  # Store the final parameters of the model
            trained_test_loss_sqrt.append(test_losses)
        trained_test_loss_sqrt = np.stack(trained_test_loss_sqrt, axis=0)
        trained_test_loss_sqrt = np.mean(trained_test_loss_sqrt, axis=0)


        x = np.arange(niter)

        # Plot the two arrays
        plt.plot(x, trained_test_losses, label="1/m", color="blue")
        plt.plot(x, trained_test_loss_sqrt, label="1/sqrt(m)", color="black")

        # Adding labels and title
        plt.xlabel('iterations')
        plt.ylabel('test-loss')
        #plt.title('Plot of Two NumPy Arrays')

        # Adding a legend
        plt.legend()
        plt.savefig('/home/bibahaduri/MVA_GDA_visuals/overfit_test_8_manual_backprop.png', dpi=300)

        breakpoint()
        pca_result = None
        # Generate loss surface in the PCA space for the last model
        grid_size = 25  # Size of the grid in the PCA space
        grid_span = 3   # Span of the grid (how far the perturbations go)
        grid_span = np.std(pca_result, axis=0) * 3
        # Assuming `student` is the final trained model, and we have X_train and Y_train
        student = StudentNetwork(m, d, )
        
        stacked_params = np.stack(final_params_all).squeeze()
        stacked_params = np.mean(stacked_params, axis=0) #pca_result.mean(axis=0)
        grid_x, grid_y, loss_values = generate_loss_surface(X_test, Y_test, student, pca_basis, stacked_params, scaling, grid_size, grid_span)

        # Plot the loss landscape and the locations of the final parameters
        plot_loss_landscape(grid_x, grid_y, loss_values, pca_result, trained_test_losses, m)

        #projections_1_over_sqrt_m = [np.dot(param - center_point, pca_basis.T) for param in final_params_1_over_sqrt_m]

        print("accomplished one")




















############################################3
# Plot Results
meana = m_ltests[-1, :, :].mean(axis=1)
meana2 = m_ltests2[-1, :, :].mean(axis=1)

plt.figure(figsize=(6, 6))
plt.semilogx(ms, meana, 'k', linewidth=2, label=r"scaling $1/\sqrt{m}$")
plt.semilogx(ms, meana2, 'gray', linewidth=2, label=r"scaling $1/m$")
plt.xlabel("m (number of neurons)")
plt.ylabel("Test loss")
plt.legend()
plt.title("Test Loss vs Number of Neurons")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()