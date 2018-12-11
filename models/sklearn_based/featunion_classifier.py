#!/bin/python
# imports
import argparse
import readdata as RD
import feature_pipeline as PPL
import numpy as np

# main
def main():
    ''' Main function to do feature-based classification
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input-file', required=True, help='The file containing text and labels (CLIN formated).')
    parser.add_argument('-t', '--test-file', required=False, help='The test data.')
    parser.add_argument('-o', '--output-dir', required=False, default='tmp', help='The directory where output is stored. Default is tmp.')
    parser.add_argument('-v', '--verbose', required=False, action='store_true', help='Verbose mode.')

    args = parser.parse_args()

    # STEP 1. Read Labels and Sentences

    if args.test_file != None:
        train_labels, train_sents = RD.readData(args.input_file)
        test_labels, test_sents = RD.readData(args.test_file)

    else:
        all_labels, all_sents=RD.readData(args.input_file)
        
        test_sents=all_sents[0:len(all_sents)//10]
        test_labels=all_labels[0:len(all_labels)//10]
        train_sents=all_sents[len(all_sents)//10:]
        train_labels=all_labels[len(all_labels)//10:]   

    # Step 2. Create the pipeline and fit a model 
    pipeline = PPL.pipeline(train_sents)
    model=pipeline.fit(train_sents, train_labels)

    # Step 3. Test he pipeline
    pred=model.predict(test_sents)

    # Step 4. Compute scores
    # Step 4.1. Accuracy:
    accuracy = np.mean(pred == test_labels)
    print(str(accuracy))

if __name__ == "__main__":
    main()
