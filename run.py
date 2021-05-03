# File to load and test models

import torch
import grammars as g
from autoencoder import SequenceLoss, evaluate, get_model, visual_eval
from grammar import GrammarGen, DataLoader, SequenceDataset, collate_batch

def main():

    LOADNAME = 'models/aEv4-bi-200-1-3.pt'
    bs = 4
    lr = 0.0001                         # Learning rate
    use_embedding = True                # Embedding Yes/No
    bidirectional = True                # bidirectional lstm layer Yes/o
    hidden_dim = 3                      # Lstm Neurons
    intermediate_dim = 200              # Intermediate Layer Neurons
    n_layers = 1                        # Lstm Layers
    dropout = 0.5
    ggen = GrammarGen()                 # Grammar
    input_dim = len( ggen )
    grammaticality_bias = 0
    punishment = 0
    loss_func = SequenceLoss( ggen, grammaticality_bias=grammaticality_bias, punishment=punishment )

    # Note: BATCH IS IN FIRST DIMENSION
    # Train
    train_seqs = ggen.stim2seqs( g.g0_train() )
    train_ds = SequenceDataset( train_seqs )
    train_dl = DataLoader( train_ds, batch_size=bs, shuffle=True, collate_fn=collate_batch )

    # Validation
    valid_seqs = ggen.generate( 12 )
    valid_ds = SequenceDataset( valid_seqs )
    valid_dl = DataLoader( valid_ds, batch_size=bs, collate_fn=collate_batch )

    # Test - Correct
    test_seqs = ggen.stim2seqs( get_correctStimuliSequence() )
    test_ds = SequenceDataset( test_seqs )
    test_dl = DataLoader( test_ds, batch_size=bs, collate_fn=collate_batch )

    # Test - Incorrect
    test_incorrect_seqs = ggen.stim2seqs( get_incorrectStimuliSequence() )
    test_incorrect_ds = SequenceDataset( test_incorrect_seqs )
    test_incorrect_dl = DataLoader( test_incorrect_ds, batch_size=bs * 2, collate_fn=collate_batch )


    ### Load Model
    model, _ = get_model( input_dim, hidden_dim, intermediate_dim, n_layers, lr, dropout, use_embedding, bidirectional )
    model.load_state_dict( torch.load( LOADNAME ) )


    ### Test
    print( '\nTrain' )
    visual_eval( model, train_dl )
    print( evaluate( model, loss_func, train_dl ) )

    print( '\nValidation' )
    visual_eval( model, valid_dl )
    print( evaluate( model, loss_func, valid_dl ) )

    print( '\nTest - Correct' )
    visual_eval( model, test_dl )
    print( evaluate( model, loss_func, test_dl ) )

    print( '\nTest - Incorrect' )
    visual_eval( model, test_incorrect_dl )
    print( evaluate( model, loss_func, test_incorrect_dl ) )


if __name__ == "__main__":
    main()