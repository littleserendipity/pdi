from Utils import Data
import Classification as cl

def main():

    ### Exemplo
    # training_data = Data().fetchFromCSV('example.csv')
    # decision_tree = cl.growTree(training_data)

    # cl.plotDiagram(decision_tree, extension="pre_prune")
    # cl.prune(decision_tree, 0.5, notify=True)
    # cl.plotDiagram(decision_tree, extension="pos_prune")
    
    # print('Classify:', cl.classify(decision_tree, [6.0, 2.2, 5.0, 1.5]))


    ### PDI
    # train_x, train_y, test_x, test_y = Data().fetchFromH5('train_catvnoncat.h5', 'test_catvnoncat.h5')
    # evaluation = "entropy"

    # training_data = cl.preProcess(train_x, train_y, normalize=True, gray=True)
    # decision_tree = cl.growTree(training_data, evaluation)

    # cl.plotDiagram(decision_tree, extension=evaluation)
    # cl.prune(decision_tree, evaluation, 0.75)
    # cl.plotDiagram(decision_tree, extension=(evaluation+"_pruned"))

    # test_data = cl.preProcess(test_x, normalize=True, gray=True)
    # result_text = cl.classify(decision_tree, test_data, test_y)
    # print("\n%s" % "\n".join(result_text))
    # Data().saveVariable(name="decision_tree", extension=(evaluation+"_classify_result"), value=result_text)


    ### CI
    train_x, train_y = Data().fetchFromPath('characters', 'font')
    test_x, test_y = Data().fetchFromPath('characters', 'hand')
    evaluation = "entropy"

    training_data = cl.preProcess(train_x, train_y, normalize=True, gray=True)
    decision_tree = cl.growTree(training_data, evaluation)

    cl.plotDiagram(decision_tree, extension=evaluation)
    cl.prune(decision_tree, evaluation, 0.5)
    cl.plotDiagram(decision_tree, extension=(evaluation+"_pruned"))

    test_data = cl.preProcess(test_x, normalize=True, gray=True)
    result_text = cl.classify(decision_tree, test_data, test_y)
    print("\n%s" % "\n".join(result_text))
    Data().saveVariable(name="decision_tree", extension=(evaluation+"_classify_result"), value=result_text)

if __name__ == '__main__':
    main()