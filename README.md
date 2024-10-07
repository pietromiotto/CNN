# Conovlutional Neural Network to calssify CIFAR10 images
Here I built, trained and tested two different models (a basic one and an upgraded one - trained on a modified dataset-) to be able to obtain an accuracy of 83% over the classification. Below you find a detailed report explaining the code and the procedures. The code is in the `CNN.ipynb` notebook, for enchanched clarity and ease of use. 

## The Problem 

The CIFAR-10 dataset is a widely used collection of images in the field
of com- puter vision. It consists of 60000 (50000 for the training and
10000 for the test) 32x32 color images across 10 different classes, with
each class containing 6,000 images. These classes include common objects
such as airplanes, automobiles, cats, and dogs. CIFAR-10 serves as a
benchmark for image classification tasks and has been instrumental in
developing and evaluating machine learning al- gorithms for image
recognition. The task of this assignment is to classify the images Note:
in the python template we set a seed. Don't change it.

## Questions 

### Data 

1.  (5 pts) Load the data (You can do it directly in PyTorch) and take
    some time to inspect the dataset. Observe at least one image per
    class and a histogram of the distribution of the images of the
    training and test set. What do you observe?

2.  (5 pts) Assume you have downloaded the dataset using the variable
    `dataset_train`. Are the entries in the correct type for the DL
    framework in PyTorch? How can you arrive at a suitable format for
    your training pipeline? Answer this question by also providing
    clarification about:

    1.  The type of each element of the dataset

    2.  How we can convert it to a suitable type. Hint: have a look at
        frame 13 of Lecture 4

    3.  The dimension of the image as a `torch.Tensor` object

    4.  The meaning of each dimension of the images

3.  (5 pts) When you arrive at this question you should have each entry
    as a `torch.Tensor` of shape (3, 32, 32) and clear the meaning of
    each dimension. A good practice in DL is to work with features
    having mean 0 and standard deviation equal to 1. Convert the dataset
    of the images in this format. To do so, you can do it from scratch
    (not recommended) or use the function
    `torchvision.transforms.Normalize`. If you go for this second
    option, don't forget that we have already transformed our dataset in
    the previous point, hence, it could be of help using the function
    `transforms.Compose`. Do not overwrite code.

4.  (5 pts) As you may have observed, we only have train and test set.
    We need a validation set for hyperparameter tuning. Create a
    validation set. Use 80% of data for the training set and 20% of the
    data for the validation set.

### Model (10 pts)

Starting from the code provided during Lecture 6, define a ConvNet. You
can only use:

-   Convolutional layers

-   Max/Avg Pooling layers

-   Activation Functions

-   Fully connected layers

For each convolutional layer you can choose padding and stride, however,
we recommend choosing padding = 0 and stride = 1. The other choices are
up to you. You can also take some inspiration from famous ConvNets.

### Training (60 pts)

1.  (15 pts) Implement the training pipeline. Make sure to code the
    following:

    -   Print and record the current training loss and accuracy every n
        steps (choose n)

    -   Print and record the current validation loss and accuracy every
        n steps (choose n)

    The validation loss will help you in hyperparameter tuning.

2.  (13 pts) Train your model. With my choice of hyperparameters, the
    best test accuracy is above 70%, hence, a necessary condition to get
    full marks is to achieve an accuracy on the test set greater than or
    equal to 70%.

3.  (2 pts) Save the parameters of the trained model as
    NAME_SURNAME_1.pt

4.  (10 pts) Plot the evolution of the train and validation loss in the
    same plot and comment on them.

5.  (18 pts) Change the architecture as you like and try to increase the
    accuracy as much as possible. Try any ideas that come to your mind
    but try to justify it. Some hints:

    -   Add Dropout (Any other hyperparameter to tune?)

    -   Change activation functions (GeLU is known to work well with
        images)

    -   Make your CNN deeper

    -   Add some regularization techniques

    -   Change the optimizer

    -   You can also use EarlyStop from Exercise 3

    -   Data augmentation

6.  (2 pts) Save the parameters of the trained model as
    NAME_SURNAME_2.pt

## Report 

### Data

1.  To load the data I create an istance of the `tochvision.Dataset`
    class for CIFAR10. For the trainset object I set `train=True`, for
    the testset I set `train=False`. As we will see afterwards, setting
    `train=True` and `train=False` allows us to use two different
    datasets, the first contains more elements than the second. With
    `classes=trainset.classes` I am saving in the variable `classes` the
    class labels of the dataset (please note that classes are the same
    both in trainset and testset). Then I create two empty arrays
    `images = []` and `labels=[]` which will contain one image for each
    class and one label for each class. Is important to note that labels
    are a numerical value, while classes are strings. Each string is
    linked to a numerical label value. To iter on elements of the
    trainset, I perform a for loop that, from each element of the
    trainset takes the image and the related label. If the label
    represents an unseen class, I save the label and the image in the
    respective arrays. The for loop stops when all 10 classes have been
    discovered. Then I create a variable `class_labels` which basically
    contains the string values for the numerical labels contained in
    `labels[]`. The method `implot` takes as arguments the `images`
    array and the `class_label` array, i.e `image[0]` would have the
    respective label in `class_label[0]`. This method basically prints a
    subplot with 2 rows and 5 columns (i.e. 2 rows containing 5 images
    of size 10x5 px). Each of these images has the respective label as
    title. As we can see from Figure
    [1](#fig:imgplot){reference-type="ref" reference="fig:imgplot"}, we
    have 10 classes: \"frog, truck, deer, automobile, bird, horse, ship,
    cat, dog, airplane\".

    ![Plot showing one image for class of CIFAR10
    dataset](images/imgplot.png)

    The function `plot_histogram` takes as arguments `classes`, the
    string label of each class, the trainset and the testset. In the
    `plot_histogram` function I first initialize two dictionaries, one
    called `train_counts={}` which will contain the occurences of each
    class label in the trainset and `test_counts={}` which will contain
    the occurences of each class label in the testset. Then iterating on
    the labels of the trainset and on the label of the testset, i fill
    these dictionary having as key `classes[label]`, so the string label
    of the class, and as value the occurence, i.e. how many times an
    image of a certain class is encountered. To clarify, in the end we
    will have a dictionary structured as (e.g.)
    `{’cat’: 10, ’deer’: 30,...}` (this is an example, it doesn't
    reflect the real dataset values). Then I plot two different
    histograms having on x axis `dictionary.keys()`, the 10 classes
    labels, and on the y axis `dictionary.values()` so the number of
    images for each class. As we can see in Figure
    [2](#fig:trainHG){reference-type="ref" reference="fig:trainHG"}, in
    the trainingset - which, as I recall, is an istance of CIFAR10
    dataset with `train=True` - we have $50.000$ images, $5000$ for each
    class. As we can see in Figure [3](#fig:testHG){reference-type="ref"
    reference="fig:testHG"} in the testset we have $10.000$ images,
    $1000$ for each class.

    ![Histogram of the distribution of images into classes of the
    training dataset](images/trainingHG.png)

    ![Histogram of the distribution of images into classes of the test
    dataset](images/testHG.png)

2.  To arrive to a suitable format for a Deep Learning task, I need to
    transform each element of the dataset into a tensor.

    -   Please note that an element `data[i]` of the dataset is a tuple
        containing an `int` (the label) and a `PIL.image.Image`, which
        is an istance of the class Image representing a 32x32 RGB image.

    -   To convert this tuple into a suitable element, when istantiating
        the object of the `torchvision.datasets.CIFAR10` class, we can
        directly specify in the argument `transform=` what kind of
        transformation to perform onto the data. If I istanciate the new
        trainset object as
        `trainset = torchvision.datasets.CIFAR10(root=’./content/CIFAR-10’, train=True,download=False, transform=transforms.ToTensor())`,
        I can obtain an istance of the class where each element is a
        tuple composed by a tensor (which is the image) and an integer
        label.

    -   the size of the image as a tensor is a 32x32 input size with
        number of channels=3.

    -   This means that each image tensor contains 3 images of 32x32,
        one for each color channel.

3.  In this step I increase the transformation operated onto the
    original dataset. To concatenate different transformation onto a
    data, I can use the function `transforms.Compose()`. This function
    allows me to perform different transformation onto the image and
    will come handful during the Data Augmentation procedure. The
    initial transformations that I apply on the CIFAR10 data are
    therefore:

    1.  `transforms.ToTensor()`: transforming the image into a
        `torch.Tensor` object

    2.  `transforms.Normalize((0,0,0), (1,1,1))])` which basically
        applies normalization to the data with mean 0 and standard
        deviation 1 for each channel. Normalization basically rescales
        the features of the data to have a mean of 0 and a standard
        deviation of 1, which helps the model to better generalize.

4.  In this part of code I create a training set containing 80% of the
    data and an evaluation set containing 20% of the data. To do so I
    define a variable `train_size=int(0.8 * len(trainset))`, which
    calculates how many samples correspond to 80% of the totality of
    train data, and a `eval_size=len(trainset) - train_size` which
    basically measures to how many samples $1-80\%$ of the training data
    corresponds. Then I call the function `torch.random_split` which
    given some inputs returns a tuple where the first element is the
    re-sized trainset and the second element is the evaluation set with
    the remaining samples. This function takes as arguments the
    trainset, the number of samples to assign to test and eval sets, and
    a `generator`. This generator is an istance of the `torch.Generator`
    class that basically generates random numbers in a reproducible way
    by specifying a seed ($42$ in this case). In this context it defines
    the randomical behaviour on which the sample will be assigned to
    trainset or evalset. In this section I also create 3 iterable
    istances, i.e. 3 batch loaders, one for the training set, one for
    the evaluation set, one for the test set. This will come handful for
    the training and testing pipeline. Batching onto the original
    dataset means to create batches or \"subsets\" of the original
    dataset on which the training pipeline can iterate. In easier words,
    with batching, the training loop, instead of iterating directly on
    each element of the dataset, it iterates on subsets of this
    datasets. Both for `trainloader`, `evalloader` and `testloader` I
    set `batch_size = 10`, the dataset is divided in batches of 10
    samples. This means that, if for the training set we have $40.000$
    samples, istead of directly iterating over the training set $40.000$
    times, we iterate $4000$ times. I choose $10$ because it speeds up
    the convergence of the model to a suitable accuracy. For
    `trainloader` I set `shuffle=True`. Enable shuffling ensures that
    each batch is a random subset of the dataset. Without shuffling,
    consecutive batches would be contiguous subsets of the data,
    potentially leading to suboptimal training. Also, for each batch
    loader I defined `num_workers=2`. Setting `num_workers` to a value
    greater than 0 will create multiple worker processes to parallelize
    data loading

### Model

My model is defined as follows:

-   **Convolutional Layers:** I created 4 convolutional layers:

    1.  The first one takes an input with three channels (since we are
        working with RGB images), and applies $32$ filters with a kernel
        size of $3x3$ to each channel. The results from each channel are
        then summed to produce a single output value for a specific
        filter.Therefore applying $32$ filters will lead to an output
        channel of $32$.

    2.  The second layer takes as input the output of the previous
        layer, so it has $32$ input channels and applies $64$ filters of
        kernel size $3x3$.

    3.  The third layer takes $64$ input channels, applies $128$ filters
        of kernel size $3x3$

    4.  The last convolutional layer takes $128$ input channels and
        applies $256$ filters of kernel size $3x3$.

-   In all convolutional layers I choose `padding=1` and a $3x3$ kernel
    size. These are pretty standard choiches. Setting `padding=1` helps
    preventing information loss at the borders of the input. Padding
    creates an \"external\" row and column filled with $0$s. The main
    reasoning behind my choiches for the Convolutional setup of this
    Neural Network, lays in this note shared by the professor of Machine
    Learning, which basically said that \" \[\...\] in CNN we have
    learnable features, where convolutional layers play the role of
    **feature detectors** and pooling layers the role of **feature
    selectors**\" My choiche of applying a high number of filters
    ($256$) was intended to allow my network to detect complex shared
    features between images. On the other hand, using a small (and
    pretty standard) kernel size ($3x3$) allows to reduce the number of
    learnable parameters (in comparison with higher kernel sizes) and
    therefore to obtain few highly complex shared feature.

-   **Pooling Layers:** I created one Max Pooling layer with $2x2$
    kernel. As I will show in the forward step, this MaxPooling layer is
    applied to the input feature map after each two convolutional
    layers. MaxPooling divides the input into non-overlapping 2x2
    regions and outputs the maximum value from each region. Thus, it
    reduces spatial dimensions, providing a downsized representation of
    the input. I choose MaxPooling instead of Average Pooling because I
    wanted to pick the most relevant shared features (also,
    AveragePooling performed worse than MaxPooling).

-   **Fully Connceted Layers:** I use 3 fully connected layers:

    1.  The first one takes as input a flattened feature map (i.e.
        $256$x$1$) and transform it into a $256 x 4$ tensor. Then
        returns a fully connected layer of output size $512$.

    2.  The second fc layer takes as input size the ouput size of the
        previous layer and returns a fully connected layer of output
        size $256$.

    3.  In the last fully connected layer the input size $256$ is
        shrunken to an output size $10$, which are the element we want
        to make predictions on (i.e. the 10 classes). These final feed
        forward layers take as input a shrunken feature map and parse it
        to make it suitable for predictions.

-   **Forward Step:** in the forward step I apply the Relu activation
    function and a pooling of $2x2$ to the output of each convolutional
    layer. The Relu activation function is a non-linear function
    composed by two linear-parts. Is used to introduce non-linearity in
    the model, i.e. to allow the model to encode non-linear
    relationships between data, and thus to describe complex non-linear
    boundaries of decision. Important feature of this activation
    function are the sparsity property and the fact that it has no
    saturation effect (i.e. it mitigates the vanishing gradient effect).
    I use it beacuse it is one of the most common choiches in
    Convolutional Neural Networks and mainly because of the sparsity
    property: since Relu returns 0 for values $<=0$, it basically
    activates only neurons that describes relevant relationship between
    data, thus it focuses on relevant features and discard the less
    informative ones (at least this is what I initially thought, as I
    will show later). I applied pooling after each convolutional layer
    in order to create, at each step, feature maps that are as
    meaningful as possible and easy to manage due to their reduced size.
    I apply relu also on each fully connected layer (except the last
    one), where having non-linearity is recommended in order to have
    more manageable values. The `torch.flatten(x,1)` shrinks the output
    of the last pooling layer preserving the channel dimension.
    Basically it allows to describe the pooled feature map in a tensor
    where each value corresponds to a specific channel in a specific
    spatial location. This makes it suitable for fully connected layers.

### Training

1.  **Training Pipeline \[lines 236-317\]:** the training pipeline is
    similar to the one i described in the previous homework (where I
    perfomed training iterating on batches). I describe it in different
    steps to make the explanation more clear:

    -   I define the DEVICE (since I used only colab, I didn't inculde
        the code for checking Apple GPU).

    -   I create an istance of the model and save it to the device.

    -   I define the loss function used. I choose
        `nn.CrossEntropyLoss()`. I choose this loss function because
        this model aims to perform a prediction task, so it is important
        to parse our results making them probabilistic. This criterion
        computes the cross entropy loss between input logits and target.

    -   I set learning rate to $0.001$. This is a small step size for
        the optimizer, but it was the best performing of the ones I
        tried. I noticed that with small batch sizes ( I am using
        `batch_size=10`), which may be linked to more noisy subsets of
        data, a small learning rate is preferable, as it tends to avoid
        getting stuck in sharp local minima and facilitates exploration
        of the parameter space.

    -   As optimizer I choose SGD, with m`momentum=0.9` and
        `weight_decay=0`, which is the default. I choose SGD beacuse, in
        this model, it outperformed other optimizers such as
        `optim.Adam` (that was one of the most suggested for this kind
        of tasks). Momentum defines what proportion of past gradients to
        take into account to the current update in backpropagation.
        $0.9$ is a high momentum, as it takes into account almost 90% of
        the past gradients.

    -   I declare two variables for storing evaluation and training loss
        at each epoch. I initialize loss with a really high value.

    -   I use $30$ epochs. With a small batchsize, the convergence is
        fast, so I don't need a lot of epochs.

    -   **Training loop on batches.** I initialize `running_loss = 0.0`
        and define the for loop onto the trainloader iterable. I perform
        a `enumerate` for loop so I can save in the variable `i` the
        step (or sub-batches) that have been considered. For each batch
        I save the inputs (images) and the labels it contains in two
        different variables, which are then saved in the DEVICE. I set
        the parameters gradients to 0 with `optimizer.zero_grad()`. I
        call the model on my inputs (in this case $10$ images) and save
        the outputs to the device. Outputs thus contains (in this case)
        $10$ estimated labels for $10$ input images. I compute the loss,
        i.e. Cross Entropy on estimated labels and real labels and then
        proceed to the backpropagation step. At each iteration I update
        the running loss, i.e. for each batch I compute the loss and add
        its value to the running loss variable. Each $2000$ batches
        considered, I print some statistics. More specifically,
        `_, predicted = torch.max(outputs, 1)` returns the predicted
        class for each input in the batch, by looking at the maximum
        value of all elements in the input tensor.
        `correct = (predicted == labels).sum().item()` returns how much
        correct predictions were made, then
        `accuracy_train = correct / batch_size` basically returns the
        number of correct predictions divided by the total number of
        element of the batch. In short, if I have a `batch_size=10`, for
        each batch I have $10$ images. `accurancy_train` stores how much
        images were correctly labeled among all the $10$ images
        contained in the batch. Lastly, I print the epoch we are in, the
        number of batches considered, the loss and the accurancy.

    -   After each epoch I append the training loss to the
        `loss_training` array.

    -   **Evaluation loop on batches.** After setting the model to
        `eval()` phase, I initialize the running loss and perform
        exactly the sam steps as for the training batch loop. The only
        noticeable difference is that in the evaluation phase we don't
        need to backpropagate, so we just compare the estimated labels
        with the real ones, compute loss and plot statistics. Monitoring
        the evaluation loss and accurancy is crucial for tuning your
        model. By looking at how it evolves you can make assumption on
        the choosen settings and how to change them to improve the model
        performance.

2.  **Compute Accurancy on the test set \[lines 321-349\]**: After the
    training and evaluation phase is concluded, I can run my model on a
    test set and compute its accuracy. The procedure is the same as for
    evaluation batch loop, but we do not need to compute loss. The final
    accuracy I get on $10.000$ unseen samples is: $72\%$. This means
    that this model, in 30 epochs returns an accuracy above $70\%$.

3.  I save the model parameters in a ph file every $2000$ batches. The
    code is the following:

                PATH = '/content/MIOTTO_PIETRO_1.ph'
                torch.save(model.state_dict(), PATH)

4.  As we can see in Figure [4](#fig:lossplot){reference-type="ref"
    reference="fig:lossplot"} the model is overfitting, because the
    training loss decreases while the evaluation loss increases. This
    means that my model is learning noise and is not generalizing enough
    to describe the relationship between samples and targets as a valid
    underlying structure. It may be worrying, but, in reality, I didn't
    take any explicit measure to reduce overfitting.

    ![Plot showing the loss evolution between training set and evauation
    set](images/Unknown-4.png)

5.  I improve my model with the following:

    Modification of the Model

    :   I modified my model taking inspiration from [this
        article](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/).
        The main difference with the old model was that I stopped
        increasing the number of filters to $128$. Also, the model is
        deeper, since it uses $6$ Convolutional Layer instead of $4$.
        This model is a so called *VGG Architecture*. VGG stands for
        Visual Geometry Group; it is a standard deep Convolutional
        Neural Network architecture with multiple layers. The "deep"
        refers to the number of layers, the most common implementations
        are VGG-16 and VGG-19, consisting of 16 and 19 convolutional
        layers. The VGG architecture is the basis of ground-breaking
        object recognition models. The VGGNet it is one of the most
        popular image recognition architectures. In this simple
        classification task the article suggested to implement a *VGG3*.
        The core of this architecture is to have two contiguous
        convolutional layers with the same output size. To clarify, the
        convolutional layers for VGG1 are defined as follows:

            `self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)`

        Thus with a VGG3 we get:

           ` self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
            self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)`

        Another difference with the old model is that `stride=1`, where
        the stride is the "step size" of the filter, i.e. "shift
        amount" between each step of the filter when it slides trough
        the image. In other words, the "stride" is the number of
        pixels the filter moves at each step. A larger stride value
        means that the filter skips more pixels when moving, resulting
        in a smaller output size and, since VGG is a deep convolutional
        network, potentially less computational cost. Consequently, the
        fully connected layers are reduced: VGG3 uses only two fully
        connected layers, the first takes as input a $128x4x4$ tensor
        and returns an output of $128$. The second reduces the input to
        $10$ predictions.

    Increase Batch Size:

    -    I increased the number of batches since with $10$ batches,
        on one side the model converges faster, but on the other its
        converges to a level of accuracy which lays around $75\%$, and
        doesn't actually get better. So I choose $32$ epochs, which
        slowed the convergence, but led to optimal results.

    Data Augmentation:
    -     I also performed data augmentation on the training set (not
        on the test set of course). Data Augmentation is performed by
        appending to `transforms.Compose()` some image transformations
        that increase the variety (and variance) of data. With more
        noisy and variegated data the model should generalize better. I
        chained together the following transformations (please look at
        Figure [5](#fig:transf){reference-type="ref"
        reference="fig:transf"} for a better understanding):

        ![Different transformation applied to perform data augmentation.
        Examples taken from [pytorch
        documentation](https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py)](images/transformations.png)

        -   `RandomHorizontalFlip()`: This transformation randomly flips
            the input image horizontally with a 50% probability. Suppose
            most of the cat images show the animale facing left. By
            applying this transformation, the dataset would consist of
            more images of cat facing right.

        -   `RandomRotation(10)`: This transformation randomly rotates
            the input image using a degree in the range $[-10; 10]$.

        -   `RandomAffine(0, shear=10, scale=(0.1, 1.2))`: to understand
            this transformation I suggest you to look at
            [5](#fig:transf){reference-type="ref"
            reference="fig:transf"}. It is pretty complicated to
            explain.

        -   `ColorJitter(brightness=0.2,contrast=0.2, saturation=0.2)`
            This transformation randomically increases/decreases
            brightness, contrast and saturation of the images.

    LeakyRelu:

  -     I used `nn.LeakyReLU(negative_slope=0.1)` which is basically a
        RELU activation function but with a negative slope, i.e. the
        negative inputs are not directly set to $0$, but they follow a
        sligthly negative slope. Basically, Leaky ReLU allows a small,
        non-zero gradient for negative inputs. I decided to use this
        activation function because, since even neurons with negative
        inputs contribute to the learning process, I wanted to not
        arbitrarily deactivate all neuron that returns values $<= 0$.

    Adding Dropout:

    -     I added dropout after each pooling layer (which in VGG is
        performed every two convolutional layers). Dropout basically
        deactivate neurons randomically, drastically reducing
        overfitting. I set `Dropout=0.5`, which means that half of the
        neurons are deactivated at each step. Deactivating neurons
        prevents the network from relying too much on specific neurons
        and forces it to learn more robust and generalized features.
        This, with data augmentation, was sufficient to reduce
        overfitting of the model (I didn't need to use weight decay).

    Adding BatchNormalization:

    -     I added Batch Normalization to each Convolutional Layer.Batch
        Normalization basically normalizes the inputs of each layer,
        speeding up the convergence. Since I increased the batch size
        (and so the time the model takes to converge), introducing batch
        normalization astonishingly sped up the process: firstly it took
        me more than $100$ epochs to reach $81\%$ accuracy, with batch
        normalization it took only $50$

    Increase Epochs:
    -     I tried different batch sizes with different epochs. With
        this learning rate I noticed that the higher the batch size, the
        slower the convergence and, up to some point, the optimal the
        results. I tried a wide variety of epochs, in increasing order.
        In Figure [6](#fig:epoch){reference-type="ref"
        reference="fig:epoch"} is it possible to see how a certain
        number of epochs led to a certain amount of accuracy (in this
        case the image refers to using `batch_size=32`).

        ![Evolution of evaluation and trainig loss, as well as evolution
        of accuracy related to increase of epochs with batch size 32.
        *Please note that the label \"step\" here refers to epochs.*
        ](images/epoch.png){#fig:epoch width="75%"}

7.  The Model parameters are saved in this line of code:

        PATH = '/content/MIOTTO_PIETRO_2.ph'
        torch.save(model2.state_dict(), PATH)
