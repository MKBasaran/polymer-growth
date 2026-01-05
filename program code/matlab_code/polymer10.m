function distribution=polymer(number_of_molecules, time_sim, p_growth, p_death, p_dead_react, kill_spawns_new, video, coloured, monomer_pool, l_exponent, d_exponent, l_naked)
%this function simulates the growth of polymers it takes;
% number_of_molecules - the number of starting chains (length 1)
% time_sim - the number of timesteps the simulation runs for
% p_growth - the probability of growth for each monomer, each time step
% p_death - the probability of a monomer dying and joining the dead pool
% p_dead_react - the probilty a dead polymer reacts with a living one
% killer_pool_size - the size of the pool of killer molecules in the system
% p_killer - the probability a killer molecule reacts with a polymer
% killer_pool_reusable - 0 means the killer pool is depleted every time a
% killer molecule is used, 1 means the killer pool is never used up
% kill_spawns_new - a binary flag as to whether a kill event means a new
% polymer of length 1 is spawned or not 1=killing event spawns a new
% polymer chain, 0 means it doesn't.
% video - binary flag on whether to output to video
% coloured - binary flag on whether to output the living, dead and coupled
% polymers as separate colours in a stacked histogram. 
% monomer_pool - the size of the monomer pool, if it is a negative number
% then no monomer pool is used and monomer supply is assumed infinite,
% p_growth is now p_growth as provided * monomer_pool size
% l_exponent - the exponent of the living length in the probability
% calculation for successful coupling
% d_exponent - the exponent of the dead length in the probability
% calculation for successful coupling
%
%
% It returns an array with the lengths of all polymers in the system
% (living and dead)

rng(2,'twister')

%Now let's write a subfunction to make the histogram
function make_histogram(living, dead, coupled, coloured, initial_monomer, current_monomer, time)
    d=[living dead coupled];
    %calculate M_n, M_w and PDI
    DPn=mean(d);
    DPw=sum(d.^2)./(DPn*size(d,2));
    PDI=DPw/DPn;
    conversion=1-current_monomer/initial_monomer;
    dlmwrite('polymerOutput.txt',[time, conversion, DPn, DPw, PDI], '-append');
    
    if coloured==0
        histogram(d,max(d)-min(d));
    else
        step=ceil((max(d)-min(d))/1000);
        binEdges=[min(d)-0.5:step:max(d)+0.5];
        living_counts=histcounts(living,binEdges);
        dead_counts=histcounts(dead,binEdges);
        coupled_counts=histcounts(coupled,binEdges);
        midbins=binEdges(1:end-1)+(binEdges(2:end)-binEdges(1:end-1))/2;
        if isempty(coupled)
            bar(midbins,[dead_counts;living_counts]','stacked')
            legend({'Dead','Living'});
        else
            bar(midbins,[coupled_counts;dead_counts;living_counts]','stacked')
            legend({'Terminated','Dead','Living'});
        end
    end
    xlabel('Length in units')
    ylabel('Frequency')
    title(strcat(' conversion=',num2str(conversion),' time=',num2str(time), ' PDI=',num2str(PDI),' DPn=',num2str(DPn),' DPw=',num2str(DPw)))
end
%setup file for output
%dlmwrite('polymerOutput.txt',{'Time','Conversion','DPn','DPw','PDI'});
file=fopen('polymerOutput.txt','w');
fprintf(file,'%s\n','Time, Conversion, DPn, DPw, PDI');
fclose(file);
%the pool of living polymers is an array of 1's of the right size
living=ones(1, number_of_molecules);
%the pool of dead polymers 
dead=[];
coupled=[];
new_dead=[];
%if we're outputing video, set it all up
if video==1
    v=VideoWriter('polymer_distribution_video.avi');
    open(v);
end
initial_monomer_pool=monomer_pool;
figure;
for t=1:time_sim
    
    if size(living,2)>0
        
        %first make a random vector with uniform random numbers which will
        %decide the fate of each living polymer
        r=rand(1, size(living,2));        
        %Now if the random number for a polymer is below p_growth, it will
        %grow.
        %%%%% LIVING GROWTH %%%%%
        if monomer_pool<0
            living(r<p_growth)=living(r<p_growth)+1;
            monomer_ratio=1;
        else
            monomer_ratio=monomer_pool/initial_monomer_pool;
            living(r<(p_growth*monomer_ratio))=living(r<(p_growth*monomer_ratio))+1;
            used_monomer=sum(r<(p_growth*monomer_ratio));
            if (used_monomer>monomer_pool)
                monomer_pool=0;
            else
                monomer_pool=monomer_pool-used_monomer;
            end
        end
        %Next if the random number for a polymer is above p_growth but below
        %p_growth+(p_death*monomer_ratio) it will die store this in new dead for now, until
        %we've worked out if any of the old dead react, We multiply by
        %monomer ratio, as we believe monomer is also involved here.
        %However, if we have infinite monomer, monomer ratio is set to 1
        %above, so then this will be just p_death)
        %%%%%%% KILLING %%%%%%%

        if kill_spawns_new==1
           % since the kill starts a new chain it uses up 1 monomer, so we
           % decrease the monomer pool
           if monomer_pool>0
                    new_dead=living(and(r<(p_growth*monomer_ratio+(p_death*monomer_ratio)),r>=p_growth*monomer_ratio));
                    living(and(r<(p_growth*monomer_ratio+(p_death*monomer_ratio)),r>=p_growth*monomer_ratio))=1;
                    number_killed=size(living(and(r<(p_growth*monomer_ratio+(p_death*monomer_ratio)),r>=p_growth*monomer_ratio)),2);
                    if number_killed>monomer_pool
                        monomer_pool=0;
                    else
                        monomer_pool=monomer_pool-number_killed;
           
                    end 
           end
        else
                new_dead=living(and(r<(p_growth*monomer_ratio+(p_death*monomer_ratio)),r>=p_growth*monomer_ratio));
                
                living(and(r<(p_growth*monomer_ratio+(p_death*monomer_ratio)),r>=p_growth*monomer_ratio))=[];
        
            
        end   
        
        %So in the new system each dead chain chooses another chain from
        %the system (either living or dead) to attack
        which_chain_attacked_per_dead = ceil(rand(1, size(dead,2))*(size(dead,2)+size(living,2)));
        
        %if the chosen chain is a dead one, we do nothing, so now only
        %consider when the chosen number is above the nunber of dead chains
        %let's implement this in a loop to consider each dead chain
        %individually
        
        still_dead=ones(1, size(dead,2));
        
        for dead_counter=1:size(dead,2)
            
            %if it chooses to attack a living chain...
            if which_chain_attacked_per_dead(dead_counter)>size(dead,2)
                which_living_attacked=which_chain_attacked_per_dead(dead_counter)-size(dead,2);
                r_success=rand();
                
                %calculate the probability of sucess given the formula from
                %Bryn

                p_success=p_dead_react/(living(which_living_attacked)^min(living(which_living_attacked)*(l_exponent/l_naked), l_exponent)*dead(dead_counter)^min(dead(dead_counter)*(d_exponent/l_naked), d_exponent));                %the dead pool
                if r_success<p_success

                    living(which_living_attacked)=living(which_living_attacked)+dead(dead_counter);
                    
                    still_dead(dead_counter)=0;
                end
            end
        end
        

        dead=dead(still_dead==1);

        
        
        %%%% OLD STUFF FROM DEAD ATTACKING IN THE OLD WAY %%%%%
            %Next look for dead polymers that will attack living ones again
        %generate a set of random numbers, this time 1 per dead polymer
%         r_dead_attack=rand(1, size(dead,2));
%         
        %if the dead_react_based_on_length flag is set then we multiply
        %these random numbers by the length of each polymer.
%         if dead_react_based_on_length==1
%             r_dead_attack=r_dead_attack.*dead;
%         end
%         %Make another random array, this time 1 number per dead polymer which
%         %will attack and then multiply it by the number of living polymers, to
%         %get an array of the indices of the living which will be attacked
%         if attack_based_on_length==1
%            %In this case shorter living polymers are more likely to be attacked 
%            %so make random number for each living polymer which is
%            %adjusted to its length
%            %so shorter polymers have lower random numbers, now pick the x
%            %lowest to be attacked
%            r_living_attacked=rand(1, size(living,2)).*living;
%            number_attacked=sum(r_dead_attack<p_dead_react);
%            [~,inds]=sort(r_living_attacked);
%            which_polymers_attacked=inds(1:number_attacked);
%         
%         else
%             which_polymers_attacked=ceil(rand(1,sum(r_dead_attack<p_dead_react))*size(living,2));
%          end
%         %Check whether any polymers were attacked
%         if size(which_polymers_attacked)>0
%             %lengthen those living polymers by the amount of the dead ones
%             %which attacked them
%             if coupled_chains_really_dead==1
%                 coupled=[coupled living(which_polymers_attacked)+dead(r_dead_attack<p_dead_react)];
%             else
%                 living(which_polymers_attacked)=living(which_polymers_attacked)+dead(r_dead_attack<p_dead_react);
%             end
%             %remove the attacking dead polymers from the pool
%             dead(r_dead_attack<p_dead_react)=[];
%         end
%         %Finally deal with the killer pool
%    %     r_killer=rand(1,killer_pool);
%     %    which_polymers_killed=ceil(rand(1,sum(r_killer<p_killer))*size(living,2));
%         %check whether some polymers are killed
%         if size(which_polymers_killed)>0
%             %increase the new dead with those killed
%             new_dead=[new_dead, living(which_polymers_killed)];
%             %delete those killed, but first check if the kill_spawns_new
%             %flag is set.
%             if kill_spawns_new==1
%                 %Reset killed polymers to length 1 in the living matrix
%                 living(which_polymers_killed)=1;
%                 if monomer_pool>0
%                     if sum(which_polymers_killed)>monomer_pool
%                         monomer_pool=0;
%                     else
%                         monomer_pool=monomer_pool-sum(which_polymers_killed);
%                     end
%                 end
%             else
%                 %Just delete them.
%                 living(which_polymers_killed)=[];
%             end
%             
%         end
%         %If the killer pool is depletable then we subtract the number of
%         %killer molecules used up from the pool.
%         if (killer_pool_reusable==0)
%             killer_pool=killer_pool-size(which_polymers_killed,2);
%         end
        %add the new dead to the dead
       dead=[dead new_dead];

      
    end
    %if we're making a video, let's get on and do it
    if video==1
        make_histogram(living, dead, coupled, coloured, initial_monomer_pool, monomer_pool,t);
        frame=getframe(gcf);
        writeVideo(v,frame);
    end
end
%return the distribution of living and dead
distribution=[living dead coupled];
%draw a histogram with 100 buckets
make_histogram(living, dead, coupled, coloured, initial_monomer_pool, monomer_pool,t);

if video==1
    close(v);
end
rand()
end