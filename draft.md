
# Quick note about backpropagation

Also, note that this is not how actual neural network gradients are implemented in practice. Libraries like PyTorch and Tensorflow do NOT explicit calculate closed form formula for the gradient. 


Instead, only the following are stored
 * partial derivative of the loss function with respect to the output of the last hidden layer. 
 * partial derivative of the nodes in the last layers with respect to the output of the second last layer.
 * so on ...
 
With those data, to evaluate a partial derivative with respect to a weight $w$ in the last layer, we use the chain rule
$$
 \frac{\partial f}{\partial w} = \sum_l \frac{\partial f}{\partial h^{(last)}_l}\frac{\partial h^{(last)}_l}{\partial w} = \frac{\partial f}{\partial h^{(last)}_{l^*}}\frac{\partial h^{(last)}_{l^*}}{\partial w}
$$
Since only one of the hidden node $l^*$ in the last layer depends on that specific weight $w$, so all other terms are 0. As mentioned before, $\frac{\partial f}{\partial h^{(last)}_{l}}$ are precomputed and stored and $\frac{\partial h^{(last)}_{l}}{\partial w}$ is just given by the derivative of the activation $\sigma$ as follow
$$
\frac{\partial h^{(last)}_{l^*}}{\partial w} = \frac{\partial}{\partial w}\sigma(wh^{last -1}_{l^*} + \text{other terms with other weights}) = \sigma'(\dots)h^{last - 1}_{l^*}. 
$$
and can be evaluated at constant time. 


If the $w$ is from earlier layers, then just expand the chain rule expression above further. 


This is reverse-mode autidifferentiation (a.k.a. backpropagation). 




### Quicknote on autodifferentiation
**Reverse mode**  
https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

If you start expanding partial derivative going from top to bottom (i.e. going from outside-in in term of function composition), then you will be doing reverse mode autodifferentiation with respect to the network inputs. Explicitly, 

$$
\begin{align*}
f_i = h^{L + 1}_i \implies 
\frac{\partial f_i}{\partial h^{L + 1}_i} &= 1 \\
\frac{\partial f_i}{\partial h^{L}_i} &= \sum_j \frac{\partial f_i}{\partial h^{L + 1}_j} \frac{\partial h^{L + 1}_j}{\partial h^{L}_i} = \frac{\partial f_i}{\partial h^{L + 1}_i} \frac{\partial h^{L + 1}_i}{\partial h^{L}_i} \tag*{since all other terms are zero} \\
\vdots \\
\frac{\partial f_i}{\partial h^{l}_i} &= \sum_j \frac{\partial f_i}{\partial h^{l + 1}_j} \frac{\partial h^{l + 1}_j}{\partial h^{l}_i} \\
\vdots \\
\frac{\partial f_i}{\partial h^{0}_i} = \frac{\partial f_i}{\partial x_i} &= \sum_j \frac{\partial f_i}{\partial h^{1}_j} \frac{\partial h^{1}_j}{\partial h^{0}_i} =\sum_j \frac{\partial f_i}{\partial h^{1}_j} \frac{\partial h^{1}_j}{\partial x_i} 
\end{align*}
$$

with 
$$
\frac{\partial h^{l + 1}_j}{\partial h^{l}_i} = \sigma'\left(a^l_i\right)W^{(l)}_{ij}
$$
where $a^l_i = \sum_j W^{(l)}_{ij} h^l_j$ being the affine transformation in each layers. 


But we don't really care about gradient with respect to network inputs do we? What we want is the gradient with respect to the **network parameters** to do parameter optimisation. We use the same top-down chain rule expansion