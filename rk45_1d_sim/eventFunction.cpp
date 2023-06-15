#include "eventFunction.h"
typedef std::vector< double > state_type;

void eventFunction(double t, state_type y)
{
    for (size_t index = 0; index < t_unit.size(); index++)
    {
        if (event_c[index] == 0)
        {
            t_unit[index] = t - phi_m[index];
            if (event_t_next[index] < t)
            {
                event_c[index] = 1;
                // Adapt phase for phase transition
                event_t_next[index] = event_t_next[index] + t_cycle;
            }
        }
        else
        {
            t_unit[index] = t - phi_c[index];
        }
        
        if (event_c[index] == 1)
        {
            if (event_t_next[index] <= t)
            {
                // Make first measurement
                //measure[index] = (y[nSize + index] + y[nSize + index + 1])/2;
                measure[index] = (y[nSize] + y[nSize + 1]) / 2;
                //cout << measure[index] << " _ _" << index << endl;
                event_c[index] = 2;
                event_t_next[index] = event_t_next[index] + (t_cycle*2);
            }
        }

        if (event_c[index] == 2)
        {
            if (event_t_next[index] <= t)
            {

                //event_data[index] = phi_c[index] % 2;

                // Calculate the score for the block
                //score_c[index] =  ((y[nSize + index] + y[nSize + index + 1]) / 2) - measure[index];
                score_c[index] = ((y[nSize] + y[nSize + 1])/2) - measure[index];


                if (score_c[index] >= score_m[index]) {
                    phi_m[index] = phi_c[index];
                }
                score_m[index] = score_c[index];
                

                
                // Overwrite the old phases with the new phase
                phi_c[index] = double(phi_m[index] + (ds_dist(gen) * ds[index]));
                // Update the next event to be the phase's transition
                event_c[index] = 0;
                event_t_next[index] = event_t_next[index] + (t_cycle * 0.9);
                //cout << event_t_next[index] << endl;
            }
        }

        // Start actuation only after phase has started
        if (event_c[index] == 3)
        {
            t_unit[0] = 0;
            //phi_c[0] = 0;
            if (phi_c[0] <= t)
            {
                event_c[index] = 1;
                event_t_next[index] = phi_c[0];
            }
        }
    }

}