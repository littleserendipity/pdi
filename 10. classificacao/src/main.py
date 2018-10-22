from Utils import Data
import Classification as cl
import Image as im

def main():

    train_x, train_y, test_x, test_y, classes = Data().fetchFromH5('train_catvnoncat.h5', 'test_catvnoncat.h5')
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    ###  [[0, 0, 0, 0, 'classe'],[],[]...[]]

    training_data = Data().fetchFromCSV('example.csv')
    # print(training_data)

    # decision_tree = cl.growTree(training_data)
    
    # cl.plotDiagram(decision_tree, extension="pre_prune")
    # cl.prune(decision_tree, 0.5, notify=False)
    # cl.plotDiagram(decision_tree, extension="pos_prune")
    
    # print('Classify:', cl.classify(decision_tree, [6.0, 2.2, 5.0, 1.5]))

if __name__ == '__main__':
    main()