### This file implements the Value, neuron, layer, and MLP classes
import math
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward        
        return out

    def __rmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + -1 * other

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + -1 * other

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(math.pow(self.data, other.data), (self, other), '**')
        def _backward():
            self.grad += other.data * math.pow(self.data, other.data - 1.) * out.grad
        out._backward = _backward   
        return out
    
    def __rpow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data ** self.data, (self, other), 'rpow')
        def _backward():
            self.grad += out.data * other.log().data * out.grad
        out._backward = _backward  
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        if other.data != 0:
            out = Value(self.data * other.data ** -1, (self, other), '/')
            return out
        else:
            print('division by zero')
            return None
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        if self.data != 0:
            return self ** -1 * other
        else:
            print('division by zero')
            return None
        
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward  
        return out

    def tanh(self):
        e2x = math.exp(2 * self.data)
        out = Value((e2x - 1.) / (e2x + 1.), (self,), 'tanh')
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        if self.data > 0:
            out = Value(math.log(self.data), (self,), 'log')
            def _backward():
                self.grad += (1 / self.data) * out.grad
            out._backward = _backward
            return out
        else:
            print('Cannot take log of a number <= 0')
            return None
        
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(_) for _ in np.random.uniform(-1, 1, nin)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((_[0] * _[1] for _ in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons] # evaluate activation of each neuron
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        size = [nin] + nouts
        self.layers = [Layer(size[_], size[_+1]) for _ in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def test():
    a = Value(4.5)
    b = Value(2.7)
    c = 3.1
    print(a+b)
    assert (a + b).data == Value(4.5 + 2.7).data
    assert (a + c).data == Value(4.5 + 3.1).data
    assert (c + a).data == Value(4.5 + 3.1).data
    assert (a - b).data == Value(4.5 - 2.7).data
    assert (a - c).data == Value(4.5 - 3.1).data
    assert (c - a).data == Value(4.5 - 3.1).data
    assert (a * b).data == Value(4.5 * 2.7).data
    assert (a * c).data == Value(4.5 * 3.1).data
    assert (c * a).data == Value(4.5 * 3.1).data
    assert (a / b).data == Value(4.5 / 2.7).data
    assert (a / c).data == Value(4.5 / 3.1).data
    assert (c / a).data == Value(3.1 / 4.5).data
    assert (a ** b).data == Value(4.5 ** 2.7).data
    assert (a ** c).data == Value(4.5 ** 3.1).data
    assert (c ** a).data == Value(3.1 ** 4.5).data
    assert (a.exp()).data == Value(math.exp(4.5)).data
    assert (a.tanh()).data == Value(math.tanh(4.5)).data
    assert (a.log()).data == Value(math.log(4.5)).data

    xs = [
    [2., 3., -1.],
    [3., -1., 0.5],
    [0.5, 1., 1.],
    [1., 1., -1.]
    ]

    ys = [1., -1., -1., 1.]

    n = MLP(3, [5, 5, 1])

    # print(ypreds - ys)

    for _ in range(100):
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        for p in n.parameters():
            p.grad = 0.0
        
        loss.backward()

        for p in n.parameters():
            p.data += -0.05 * p.grad
        print(f'Loss: {loss.data}')
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
    print('Final loss:', loss.data)
    print('Final preds:', ypred)

def main():
    test()

if __name__ == '__main__':
    main()



# class Neuron:
#     def __init__(self, data, children=(), label=''):
#         self.data = data
#         self._prev = children
#         self.label = label
#         self.grad = 0.0
    
#     def __call__(self, x):
#         self.data = 