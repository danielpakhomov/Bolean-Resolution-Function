from asyncio import current_task
from email import policy
import random
import numpy as np
def get_states():
    all_states=[]
    for i in range(4):
        for j in range(4):
                all_states.append((i,j))
    return all_states
def setrewards(all_states,vState, fState, victoryR, failureR):
    rewards = {}
    for i in all_states:
        if i == fState:
            rewards[i] =failureR
        elif i ==vState:
            rewards[i] = victoryR
        else:
            rewards[i] = 0
    return rewards 
def getactions(all_states,fState,vState):
    actions1={}
    for i in all_states:
        if i==fState or i ==vState:
            continue
        elif i==(0,0) or i==(3,0) or i==(0,3) or i==(3,3):
            if i==(0,0):
                actions1[i]=('D','R')
            elif i==(0,3):
                actions1[i]=('D','L')    
            elif i==(3,0):
                actions1[i]=('U','R')    
            else:
                actions1[i]=('U','L')
        elif i==(0,1) or i==(0,2):
            actions1[i]=('D','L','R')
        elif i==(1,0) or i==(2,0):
            actions1[i]=('D','U','R')
        elif i==(3,1) or i==(3,2):
            actions1[i]=('U','R','L')
        elif i==(1,3) or i==(2,3):
            actions1[i]=('D','L','U')
        else:
            actions1[i]=('D', 'R', 'L', 'U') 
    return actions1           

def value_iteration(noiseP, vState, fState, victoryR, failureR, gamma, max_iternum):
    vState=tuple(vState)
    fState=tuple(fState)
    theta = 1e-10
    all_states=get_states()
    rewards = setrewards(all_states,vState, fState, victoryR, failureR)
    actions=['L','R','U','D']
    actions1=getactions(all_states,fState,vState)
    policy={}
    for s in actions1.keys():
        policy[s] = np.random.choice(actions1[s])

    #initial value function 
    V={}
    for s in all_states:
        if s in actions1.keys():
            V[s] = 0
        if s==fState:
            V[s]=failureR
        if s==vState:
            V[s]=victoryR        

    for iter in range(max_iternum):
        delta=0
        for s in all_states:            
            if s in policy:
                
                old_v = V[s]
                new_v = 0
                
                for a in actions1[s]:
                    act=[]
                        
                    if a == 'U':
                        nxt = [s[0]-1, s[1]]
                        if 'L' in actions1[s]:
                            act.append((s[0],s[1]-1))
                        elif 'L' not in actions1[s]:
                            act.append((s[0]-1,s[1]))    
        
                        if 'R' in actions1[s]:
                            act.append((s[0],s[1]+1))
                        elif 'R' not in actions1[s]:
                            act.append((s[0]-1,s[1]))        
                    if a == 'D':
                        nxt = [s[0]+1, s[1]]
                        if 'L' in actions1[s]:
                            act.append((s[0],s[1]-1))
                        elif 'L' not in actions1[s]:
                            act.append((s[0]+1,s[1]))    
        
                        if 'R' in actions1[s]:
                            act.append((s[0],s[1]+1))
                        elif 'R' not in actions1[s]:
                            act.append((s[0]+1,s[1]))
                    if a == 'L':
                        nxt = [s[0], s[1]-1]
                        if 'U' in actions1[s]:
                            act.append((s[0]-1,s[1]))
                        elif 'U' not in actions1[s]:
                            act.append((s[0],s[1]-1))    
        
                        if 'D' in actions1[s]:
                            act.append((s[0]+1,s[1]))
                        elif 'D' not in actions1[s]:
                            act.append((s[0],s[1]-1)) 
                    if a == 'R':
                        nxt = [s[0], s[1]+1]
                        
                        if 'U' in actions1[s]:
                            act.append((s[0]-1,s[1]))
                        elif 'U' not in actions1[s]:
                            act.append((s[0],s[1]+1))    
        
                        if 'D' in actions1[s]:
                            act.append((s[0]+1,s[1]))
                        elif 'D' not in actions1[s]:
                            act.append((s[0],s[1]+1))

                    nxt = tuple(nxt)
                    tr_nxt=0
                    # act = tuple(act)
                    if nxt not in act:
                        tr_nxt=(1-noiseP)* V[nxt]
                    elif nxt in act:
                        tr_nxt=(1-noiseP/2)*V[nxt]
                    for ac in act:
                        if ac!=nxt:
                            tr_nxt+=(noiseP/2)*V[ac]

                    v = rewards[s] + (gamma * (tr_nxt))
                    if v > new_v: 
                        new_v = v
                        policy[s] = a

                                        
                V[s] = new_v
                delta = max(delta,np.abs(old_v - V[s]))
        
        if delta<theta:
            break    
                    
                
    UValue=[val for k,val in V.items()]
    p=0
    UValues=[]
    while(p<len(UValue)):
        Uval=[]
        for i in range(p,p+4):
            if(UValue[i]==failureR) or (UValue[i]==victoryR):
                UValue[i]=0
            Uval.append(UValue[i])
        UValues.append(Uval)
        p=p+4
    return UValues


def select_action(Q,current_state,epsilon,actions1):
    n=random.random()
    action='X'
    if n<epsilon:
        action=random.choice(actions1)
    else:
        if(len(set(Q[current_state]))==1):
            action=random.choice(actions1)
        else:
            maxi=max(Q[current_state])
            indx=Q[current_state].index(maxi)
            if(indx==0):
                action='L'
            elif(indx==1):
                action='R'
            elif(indx==2):
                action= 'U'
            elif(indx==3):
                action='D'                 
      
    return action 

def getmaxQ(Q,current_state,actions1,fState,vState):
    
    if current_state==fState or current_state==vState:
        return 0
    else:    
        maxi=max(Q[current_state])
      
    return maxi                    


def QL_explore(noiseP, initS0, vState, fState, victoryR, failureR, gamma, alpha,
epsilon, max_iternum):
    policy=[]
    vState=tuple(vState)
    fState=tuple(fState)
    for i in range(4):
        pol=[]
        for j in range(4):
            pol.append('X')
        policy.append(pol)    
    all_states=get_states()
    Q_values1={}
    for i in all_states:
        q_va=[]
        for j in range(4):
            q_va.append(0)
        Q_values1[i]=q_va
     
    actions1=getactions(all_states,fState,vState) 
    actions=['L','R','U','D']
    rewards =setrewards(all_states,vState, fState, victoryR, failureR)   
    for iter in range(max_iternum):
        current_state=tuple(initS0)
        while True:
            action=select_action(Q_values1,current_state,epsilon,actions)
    
            nxt=()
            ac=5
            if action=='U':
                nxt=(current_state[0]-1, current_state[1])
                ac=2
            if action=='D':
                nxt=(current_state[0]+1, current_state[1])
                ac=3
            if action=='L':
                nxt=(current_state[0], current_state[1]-1)
                ac=0
            if action=='R':
                nxt=(current_state[0], current_state[1]+1)
                ac=1
            if(nxt[0]>3 or nxt[0]<0 or nxt[1]>3 or nxt[1]<0):
                nxt=current_state    
            old_Q=Q_values1[current_state][ac]
            delta_Q=rewards[nxt]+gamma*getmaxQ(Q_values1,nxt,actions,fState,vState)-Q_values1[current_state][ac]
            Q_values1[current_state][ac]+=alpha*delta_Q
            if(old_Q<=Q_values1[current_state][ac]):
                policy[current_state[0]][current_state[1]]=action
            
            current_state=nxt
            if nxt==vState or nxt==fState:
                    break
    return Q_values1,policy                
                






