# CS230

## data
        from sklearn.datasets import fetchopenml
        mnist = fetch_openml('mnist_784', version=1)
        
## initialize Network
        
        BGD = [NetworkLayers(layer=1, neurons=200), NetworkLayers(layer=2, neurons=50),
            NetworkLayers(layer=3, neurons=10, final=True)]
            
        Adam = [AdamNetwork(layer=1, neurons=200), AdamNetwork(layer=2, neurons=50),
            AdamNetwork(layer=3, neurons=10, final=True)]
            
        RMS = [RMSNetwork(layer=1, neurons=200), RMSNetwork(layer=2, neurons=50),
            RMSNetwork(layer=3, neurons=10, final=True)]
            
        Momentumn = [MomentumNetwork(layer=1, neurons=200), MomentumNetwork(layer=2, neurons=50),
            MomentumNetwork(layer=3, neurons=10, final=True)]

- NOTES: you can add more layers in the above `network` list, just remember that set the `final=True`
of the last layer

### set hyper-parameters
- the list of important hyper-parameters `lambd`, `beta`, `alpha(learning rate)`, `beta2`
- just set as `Network(..., hyper-parameter=value, ...)` when initializing networks

## train model
- run `Models.build_model(parameters)`
- please remember to set `model=Momentum, RMS or Adam` according to the network you choose, if not, its default `gd` for BGD

## NOTES
- in `RMS` and `Adam`, the learning rate must be small. otherwise, the model can not be trained correctly 
and have the cost `nan` problem. the learning rate is `0.0001` in my testing

## Conclusion
- `RMS` and `Adam` can overfit the train dataset easier, their hyper-parameters and parameters need to be tuned carefully
- in the final layer, `error= output - y`, which is the same as that of MSE by chance,
 but actually its loss function should be `loss=y*log(output) + (1-y)*log(1-output)` for binary classification,
 `loss=y*log(output)` for multi-class classification
## Next step
- add `Batch Normalization` to this application (5/19/2020 updated)
