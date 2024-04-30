epochs = 250
batch_size = 12
learning_rate = 0.003
alpha_plus = 0.03       #learning rate for positive errors
alpha_minus = 0.07      #learning rate for negative errors
# alpha_minus = 1
# alpha_plus = 1

dataloader = make_dataloader(E, batch_size=batch_size)
scores, other_scores, weights = train_model(model, dataloader, 
                                            epochs=epochs, 
                                            optimizer='SGD', 
                                            lr=learning_rate,
                                            alpha_plus = alpha_plus,
                                            alpha_minus = alpha_minus,
                                            print_every=25,
                                            use_weights='final')

plt.figure(figsize=(3,3), dpi=300), plt.plot(scores), plt.xlabel('# epochs'), plt.ylabel('loss')