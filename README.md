# Neural-Persona-based-Conversation-Model-Python-Version

This code is a PyTorch re-implementation of the persona-based neural conversation model proposed by Jiwei Li, Michel Galley, Chris Brockett, Georgios P. Spithourakis, Jianfeng Gao, Bill Dolan:

J.Li, M.Galley, C.Brockett, J.Gao and B.Dolan. "[A persona-based neural conversation model](https://arxiv.org/pdf/1603.06155.pdf)". ACL2016.

Thank them for their great contributions :)

The code was used in our project of "Automatic Evaluation of Neural Personality-based Chatbot":

Y.Xing, R.Fernández. "[Automatic Evaluation of Neural Personality-based Chatbot](https://arxiv.org/pdf/1810.00472.pdf)". INLG 2018.

# Download Data

Please go to "https://github.com/jiweil/Neural-Dialogue-Generation" for the processed data provided by Jiwei Li.

# Training

Similar to the original Lua codes, run

    python train.py --(optional parameter)

The parameters could be checked in train.py.

After training, trained models are stored in /save/testing.

# Decode

Similar to the original Lua codes, run

    python decode.py --(optional parameter)

The parameters could be checked in decode.py.

After decoding, the outputs are stored in /outputs.

# Reference

Please refer to https://github.com/jiweil/Neural-Dialogue-Generation for more details.
