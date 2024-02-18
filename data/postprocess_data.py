from modelzipper.tutils import *
import numpy as np
import pdb

results = auto_read_data(FILE)



def postprocess(x):
    """
    x: batch x seq_len x 9
    """
    # first remove the 1, 2 columns
    m_x = torch.cat((x[:, :1], x[:, 3:]), dim=1)

    # find the right command value
    m_x[:, 0] = torch.round(m_x[:, 0] / 100) * 100

    # clip all the value to max bins 
    m_x = torch.clamp(m_x, 0, 200)

    # process the M and L path
    m_x[:, 1:5][m_x[:, 0] != 200] = 0

    # add to extra column to satify the 9 columns
    x_0_y_0 = torch.zeros((m_x.size(0), 2), dtype=m_x.dtype)
    x_0_y_0[1:, 0] = m_x[:-1, -2]  # x_3 of the previous row
    x_0_y_0[1:, 1] = m_x[:-1, -1]  # y_3 of the previous row
    full_x = torch.cat((m_x[:, :1], x_0_y_0, m_x[:, 1:]), 1)
    return full_x


for batch in results:
    for i in range(len(batch['predict'])):
        predict, golden = batch['predict'][i], batch['input'][i]
        p_predict = postprocess(predict)

