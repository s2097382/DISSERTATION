import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby, product
import time
import gurobipy as gp
from gurobipy import GRB
from intersect import intersection

#Data file
data = "Test Files/Veri_5.in"

def run(data,obj):
    #Import data
    f = open(data, "r")
    lines = f.readlines()
    f.close()

    lines = [x for x in lines if x != '\n']
    #convery numbers in data to float
    # split_dat = [x[:len(x)-1].split(',') for x in lines[::2]] #for test data
    split_dat = [x[:len(x)-1].split(',') for x in lines]

    def maybe_float(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return s
    data_int = [[maybe_float(i) for i in row] for row in split_dat]

    #extract hourly data
    hourly_data = [x for x in data_int if "S" in x ]
    block_data = [x for x in data_int if "B" in x ]
    flexible_data = [x for x in data_int if "F" in x ]

    #list of all hourly bids and time period they are in
    hourly_bids = sorted(list(set([(x[0],x[2]) for x in hourly_data])))
    #list of block bids
    block_bids = sorted(list(set([x[0] for x in block_data])))
    #list of flexible bids
    flexible_bids = sorted(list(set([x[0] for x in flexible_data])))

    #get supply and demand bids
    hourly_supply_bids = list(set([(x[0],x[2]) for x in hourly_data if x[4] < 0]))
    hourly_demand_bids = list(set([(x[0],x[2]) for x in hourly_data if x[4] > 0]))
    block_supply_bids = [x[0] for x in block_data if x[4] < 0]
    block_demand_bids = [x[0] for x in block_data if x[4] > 0]
    flex_supply_bids = [x[0] for x in flexible_data if x[4] < 0]
    flex_demand_bids = [x[0] for x in flexible_data if x[4] > 0]

    #get hourly supply and demand data
    hourly_supply_data = [ x for x in hourly_data if x[0] in [i[0] for i in hourly_supply_bids] ]
    hourly_demand_data = [ x for x in hourly_data if x[0] in [i[0] for i in hourly_demand_bids] ]

    #parent and child block bids
    child_bids = [x for x in block_data if x[7] != ""]
    child_supply_bids = [x[0] for x in child_bids if x[4] < 0]
    child_demand_bids = [x[0] for x in child_bids if x[4] > 0]

    # parent_bids = []
    # linked_bids = {} #parent bid: each child bid that is linked
    # for x in child_bids:
    #     parent = x[7]
    #     if parent in linked_bids:
    #         linked_bids[parent].append(x[0])
    #     else:
    #         linked_bids[parent] = [x[0]]
    #
    #     if parent not in parent_bids:
    #         parent_bids.append(x[7])
    parent_bids = []
    child_parent = []
    for x in child_bids:
        parent = x[7]
        child_parent.append([x[0],parent])
        if parent not in parent_bids:
            parent_bids.append(x[7])


    #code for plots contained here

    def plots(h, print_plot):


        h_sup = [x for x in hourly_supply_data if x[2] == h ]
        h_dem = [x for x in hourly_demand_data if x[2] == h ]

        #unique supply bids
        used_bids = set()
        sup_bids = [x[0] for x in h_sup if x[0] not in used_bids and (used_bids.add(x[0]) or True)]
        #unique demand bids
        used_bids = set()
        dem_bids = [x[0] for x in h_dem if x[0] not in used_bids and (used_bids.add(x[0]) or True)]

        #list of all price break points
        used = set()
        sup_bp = sorted([float(x[5]) for x in h_sup if x[5] not in used and (used.add(x[5]) or True)])
        used = set()
        dem_bp = sorted([float(x[5]) for x in h_dem if x[5] not in used and (used.add(x[5]) or True)])

        sup_qp_pair = []
        for b in sup_bids:
            dat = [x for x in h_sup if x[0] == b ] #data for this bid
            b_price = [x[5] for x in h_sup if x[0] == b]
            b_quantity = [abs(x[4]) for x in h_sup if x[0] == b]

            #for each break point check if that bp exists for current bid
            #if not then find quantity and add to list
            for i in sup_bp:
                if i in b_price: #if this bp already in bid then add
                    sup_qp_pair.append([b_quantity[b_price.index(i)],i])
                else:
                    sup_qp_pair.append([np.interp(i, b_price,b_quantity),i])


        dem_qp_pair = []
        for b in dem_bids:
            dat = [x for x in h_dem if x[0] == b ] #data for this bid
            b_price = [x[5] for x in h_dem if x[0] == b]
            b_quantity = [abs(x[4]) for x in h_dem if x[0] == b]

            #for each break point check if that bp exists for current bid
            #if not then find quantity and add to list
            for i in dem_bp:
                if i in b_price: #if this bp already in bid then add
                    dem_qp_pair.append([b_quantity[b_price.index(i)],i])
                else:
                    dem_qp_pair.append([np.interp(i, b_price,b_quantity),i])



        #horizontally add all demand together for each price break points
        f=lambda x: x[1]
        agg_sup = sorted([[sum(x[0] for x in g),k] for k, g in groupby(sorted(sup_qp_pair, key=f), key=f)], key=lambda x: x[1], reverse=False)
        agg_dem = sorted([[sum(x[0] for x in g),k] for k, g in groupby(sorted(dem_qp_pair, key=f), key=f)], key=lambda x: x[1], reverse=False)

        sup_q = [x[0] for x in agg_sup]
        sup_p = [x[1] for x in agg_sup]
        dem_q = [x[0] for x in agg_dem]
        dem_p = [x[1] for x in agg_dem]



        if print_plot:
            plt.figure(figsize=(10,8))
            # Plot the x and y values on the graph
            # plt.plot(sup_quantity,sup_price,'r',label="Supply", marker='.')
            # plt.plot(dem_quantity,dem_price,'b',label="Demand")
            plt.plot(sup_q,sup_p,'r',label="Supply", marker='.')
            plt.plot(dem_q,dem_p,'b',label="Demand")
            plt.xlabel("Quantity (MWh)")
            plt.ylabel("Price (TL/MWh)")
            plt.title("Hour "+str(h))
            plt.legend(loc="upper left")
            plt.show() # Display the graph



        a,b = intersection(dem_q, dem_p, sup_q, sup_p)

        #dont know why
        if len(a) > 1:
            a,b = a[0],b[0]


        inter = [float(a),float(b)]

        return inter,agg_sup,agg_dem


    #-------------------------------------------------------------------------------
    ##Initialise parameters


    #set of segments (assume same for each hour)
    segments = len([x for x in hourly_data if x[0] == hourly_bids[0][0]]) - 1
    L_i = list(range(1,segments+1)) #set of segments (same for each hour)

    #list of time periods
    Z = list(set([x[2] for x in hourly_data]))

    #Calculate F_z for supply and demand
    #We take this as the first price entry for each bid since the demand bids start at the highest quantity anf work backwards
    F_z = {}
    for z in Z:
        #get min and max price of bids per hour
        #only need to use supple as the bp for supply and demand are the same by definition
        inter,agg_sup,agg_dem = plots(z,False)
        F_z[z] = [agg_sup[0][1],agg_dem[0][1]]


    #### NOTE
    # Data imputs demand from last point at start
    # But segments start ffrom 0 so we need to accuont for this

    ## First and last prices for lth segment of hourly bid i in time z
    first_price = {} #Dictionary of form (bid,time period, segment): first price of segment
    last_price = {} #Dictionary of form (bid,time period, segment): last price of segment
    first_quantity = {} #Dictionary of form (bid,time period, segment): first quantity of segment
    last_quantity = {} #Dictionary of form (bid,time period, segment): last quantity of segment

    #populate dictionaries
    for i in hourly_bids:
        bid_info = [x for x in hourly_data if x[0] == i[0]]

        for x in bid_info:
            if x[1] == len(L_i)+1: #if last segment
                last_price[x[0],x[2],previous[1]] = x[5]
                last_quantity[x[0],x[2],previous[1]] = abs(x[4]) ##CHECK IF NEED ABSOLUTE VALUE

            elif x[1] == 1: #if first segment
                first_price[x[0],x[2],x[1]] = x[5]
                first_quantity[x[0],x[2],x[1]] = abs(x[4])

            else:
                last_price[x[0],x[2],previous[1]] = x[5]
                last_quantity[x[0],x[2],previous[1]] = abs(x[4])

                first_price[x[0],x[2],x[1]] = x[5]
                first_quantity[x[0],x[2],x[1]] = abs(x[4])

            previous = x


    #get all combinations of izl
    combinations,_ = gp.multidict(first_price)

    ##Parameters for block bids

    block_price = {} #price of each bid
    block_quantity = {} #quantity of each bid
    hours_covered = {} #hours covered by each bid
    N = {} #amount of hours convered by bid

    for b in block_bids:
        bid_info = [x for x in block_data if x[0] == b][0]
        block_price[bid_info[0]] = bid_info[5]
        block_quantity[bid_info[0]] = abs(bid_info[4])
        hours_covered[b] = list(range(int(bid_info[2]), int(bid_info[2])+int(bid_info[6])))
        N[b] = len(hours_covered[b])


    #disdtonary delta - delta{bid,time period} = 1 if bid spans that period
    delta = {}
    for b in block_bids:
        for z in Z:
            if z in hours_covered[b]:
                delta[b,z] = 1
            else:
                delta[b,z] = 0

    gamma = 100000


    ## parameters for flexible bids
    flex_price = {}
    flex_quantity = {}

    for f in flexible_bids:
        bid_info = [x for x in flexible_data if x[0] == f][0]
        flex_price[bid_info[0]] = bid_info[5]
        flex_quantity[bid_info[0]] = abs(bid_info[4])

    flex_comb = list(product(flexible_bids, Z))
    flex_sup_comb = [x for x in flex_comb if x[0] in flex_supply_bids]
    flex_dem_comb = [x for x in flex_comb if x[0] in flex_demand_bids]

    ###******************************
    ###******************************

    #to supres output
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()


    ### Create LP model
    m = gp.Model('FlexBids', env=env)

    # decision variables
    x = m.addVars(combinations,vtype=GRB.CONTINUOUS, name="fraction")
    y = m.addVars(len(block_bids),vtype=GRB.BINARY, name="block")
    f = m.addVars(len(Z), vtype=GRB.CONTINUOUS, name="MCP")
    vv = m.addVars(flex_comb, vtype=GRB.BINARY, name = "flexible")
    w = m.addVars(combinations, vtype=GRB.BINARY, name = "Auxiliary")


    ### constraints
    #suppy demand balance constraint -  did not sum over z as the hourly bid is already linked to hour
    # c2 = m.addConstr( ( gp.quicksum( (last_quantity[(i[0],i[1],l)]-first_quantity[(i[0],i[1],l)])*x[i[0],i[1],l] for i in hourly_supply_bids for l in L_i)
    #                             - gp.quicksum( (last_quantity[(i[0],i[1],l)]-first_quantity[(i[0],i[1],l)])*x[i[0],i[1],l] for i in hourly_demand_bids for l in L_i)
    #                             + gp.quicksum( first_quantity[(i[0],i[1],1)] for i in hourly_supply_bids )
    #                             - gp.quicksum( first_quantity[(i[0],i[1],1)] for i in hourly_demand_bids )
    #                              == 0 ), name = 'balance')

    for z in Z:
        s_bids = [x[0] for x in hourly_supply_bids if x[1] == z]
        d_bids = [x[0] for x in hourly_demand_bids if x[1] == z]
        c2 = m.addConstr( ( gp.quicksum( (last_quantity[(i,z,l)]-first_quantity[(i,z,l)])*x[i,z,l] for i in s_bids for l in L_i)
                                - gp.quicksum( (last_quantity[(i,z,l)]-first_quantity[(i,z,l)])*x[i,z,l] for i in d_bids for l in L_i)
                                + gp.quicksum( first_quantity[(i,z,1)] for i in s_bids )
                                - gp.quicksum( first_quantity[(i,z,1)] for i in d_bids )
                                + gp.quicksum( delta[b,z] * block_quantity[b] * y[block_bids.index(b)] for b in block_supply_bids)
                                - gp.quicksum( delta[b,z] * block_quantity[b] * y[block_bids.index(b)] for b in block_demand_bids)
                                + gp.quicksum( flex_quantity[e] * vv[e,z] for e in flex_supply_bids)
                                - gp.quicksum( flex_quantity[e] * vv[e,z] for e in flex_demand_bids)
                                 == 0 ), name = 'balance'+str(z))

    #hourly constraints
    c3comb = [x for x in combinations if x[2] == 1]
    c3a = m.addConstrs((w[(i,z,l)] <= x[(i,z,l)] for i,z,l  in c3comb ), name = 'c3a')
    c3b = m.addConstrs((x[(i,z,l)] <= 1 for i,z,l  in c3comb ), name = 'c3b')

    c4comb = [x for x in combinations if x[2] in range(2,len(L_i))]
    c4a = m.addConstrs((w[(i,z,l)] <= x[(i,z,l)] for i,z,l  in c4comb ), name = 'c4a' )
    c4b = m.addConstrs((x[(i,z,l)] <= w[(i,z,l-1)] for i,z,l  in c4comb ), name = 'c4b' )

    c5comb = [x for x in combinations if x[2] == len(L_i)]
    c5a = m.addConstrs((0 <= x[(i,z,l)] for i,z,l  in c5comb ), name = 'c5a')
    c5b = m.addConstrs((x[(i,z,l)] <= w[(i,z,l-1)] for i,z,l  in c5comb ), name = 'c5b')

    c6 = m.addConstrs( (gp.quicksum( (last_price[(i[0],i[1],l)]-first_price[(i[0],i[1],l)])*x[i[0],i[1],l] for l in L_i )
                          + F_z[i[1]][0]  == f[i[1]-1] for i in hourly_supply_bids)  , name = 'c6')

    c7 = m.addConstrs( (gp.quicksum( (last_price[(i[0],i[1],l)]-first_price[(i[0],i[1],l)])*x[i[0],i[1],l] for l in L_i )
                            + F_z[i[1]][1] == f[i[1]-1] for i in hourly_demand_bids)  , name = 'c7')
    #
    # #Constraints without F_z
    # c6 = m.addConstrs( (gp.quicksum( (last_price[(i[0],i[1],l)]-first_price[(i[0],i[1],l)])*x[i[0],i[1],l] for l in L_i )
    #                         == f[i[1]-1] for i in hourly_supply_bids)  , name = 'c6')
    #
    # c7 = m.addConstrs( (gp.quicksum( (last_price[(i[0],i[1],l)]-first_price[(i[0],i[1],l)])*x[i[0],i[1],l] for l in L_i )
    #                          == f[i[1]-1] for i in hourly_demand_bids)  , name = 'c7')


    #BLOCK CONSTRAINTS
    #C8 - SUPPLY - If price of bid is less than average MCP over time periods then accept
    c8 = m.addConstrs(  (-N[b]*block_price[b] + gp.quicksum( delta[b,z] * f[z-1] for z in Z ) <= gamma * y[block_bids.index(b)] for b in block_supply_bids if b not in child_supply_bids) , name = 'c8')
    # DEMAND - If bid price is greater than average price the accept
    ### HAVE CHANGED THE SIGN FOR THIS ONE
    c9 = m.addConstrs(  (N[b]*block_price[b] - gp.quicksum( delta[b,z] * f[z-1] for z in Z ) <= gamma * y[block_bids.index(b)] for b in block_demand_bids if b not in child_demand_bids) , name = 'c9')


    #C10

    c10 = m.addConstrs( (y[block_bids.index(b)] <= y[block_bids.index(k)] for b,k in child_parent)       , name = 'c10')

    #FLEXIBLE CONSTRAINTS

    c11 = m.addConstrs((gp.quicksum( vv[(e,z)] for z in Z ) <= 1 for e in flexible_bids), name = 'c11')

    c12 = m.addConstrs( ( f[z-1] - flex_price[e] <= gamma * gp.quicksum( vv[(e,k)] for k in Z ) for e,z in flex_sup_comb), name = 'c12')

    c13 = m.addConstrs( ( flex_price[e] - f[z-1] <= gamma * gp.quicksum( vv[(e,k)] for k in Z ) for e,z in flex_dem_comb), name = 'c13')

    #Objective function Parameters



    total_area_demand = {}
    all_demand_area = 0
    for i in hourly_demand_bids:
        area = 0
        for l in L_i:
            area += 0.5*(2*first_price[(i[0],i[1],l)] +  (last_price[(i[0],i[1],l)]-first_price[(i[0],i[1],l)])) * (first_quantity[(i[0],i[1],l)]-last_quantity[(i[0],i[1],l)])

        area += last_quantity[(i[0],i[1],L_i[-1])] * last_price[(i[0],i[1],L_i[-1])] #add in area until y axis
        total_area_demand[i[0],i[1]] = area
        all_demand_area += area


    #block obj
    block_supply_obj = gp.quicksum( N[b]*block_quantity[b]*block_price[b]*y[block_bids.index(b)]   for b in block_supply_bids)
    block_demand_obj = gp.quicksum( N[b]*block_quantity[b]*block_price[b]*y[block_bids.index(b)]   for b in block_demand_bids)

    #flexible obj

    flex_demand_obj = gp.quicksum( flex_quantity[e]*flex_price[e] * gp.quicksum(vv[e,z] for z in Z) for e in flex_demand_bids)
    flex_supply_obj = gp.quicksum( flex_quantity[e]*flex_price[e] * gp.quicksum(vv[e,z] for z in Z) for e in flex_supply_bids)


    #extra term to account for initial values
    #all bids
    extra_term = gp.quicksum( first_quantity[(i[0],i[1],1)] - first_price[(i[0],i[1],1)] for i in hourly_bids)
    #demand bids
    # extra_term = gp.quicksum( first_quantity[(i[0],i[1],1)] - first_price[(i[0],i[1],1)] for i in hourly_demand_bids)

    ## OBJ1 - Maximise market surplus
    #hourly demand area = C + B

    if obj == 1:

        hourly_demand_area = gp.quicksum( (total_area_demand[(i[0],i[1])]) for i in hourly_demand_bids ) - gp.quicksum( 0.5*(2*first_price[(i[0],i[1],l)] +  (last_price[(i[0],i[1],l)]-first_price[(i[0],i[1],l)])*x[i[0],i[1],l]) * abs(last_quantity[(i[0],i[1],l)]-first_quantity[(i[0],i[1],l)])*x[i[0],i[1],l] for i in hourly_demand_bids for l in L_i )
        hourly_supply_area = gp.quicksum( 0.5*(2*first_price[(i[0],i[1],l)] +  (last_price[(i[0],i[1],l)]-first_price[(i[0],i[1],l)])*x[i[0],i[1],l]) * (last_quantity[(i[0],i[1],l)]-first_quantity[(i[0],i[1],l)])*x[i[0],i[1],l] for i in hourly_supply_bids for l in L_i )

        m.setObjective(hourly_demand_area - hourly_supply_area + block_demand_obj - block_supply_obj + flex_demand_obj - flex_supply_obj , GRB.MAXIMIZE)

    elif obj == 2:

        ##OBJ 2
        # #demand area is area under demand curve starting from RHS (A)
        hourly_demand_area =  gp.quicksum( 0.5*(2*first_price[(i[0],i[1],l)] +  (last_price[(i[0],i[1],l)]-first_price[(i[0],i[1],l)])*x[i[0],i[1],l]) * abs(last_quantity[(i[0],i[1],l)]-first_quantity[(i[0],i[1],l)])*x[i[0],i[1],l] for i in hourly_demand_bids for l in L_i )
        hourly_supply_area = gp.quicksum( 0.5*(2*first_price[(i[0],i[1],l)] +  (last_price[(i[0],i[1],l)]-first_price[(i[0],i[1],l)])*x[i[0],i[1],l]) * (last_quantity[(i[0],i[1],l)]-first_quantity[(i[0],i[1],l)])*x[i[0],i[1],l] for i in hourly_supply_bids for l in L_i )

        m.setObjective(hourly_demand_area + hourly_supply_area - block_demand_obj + block_supply_obj - flex_demand_obj + flex_supply_obj, GRB.MINIMIZE)

    elif obj == 3:
        #obj3 - min average price
        m.setObjective( (gp.quicksum(f[z] for z in range(len(Z))))/len(Z) ,GRB.MINIMIZE)


    # Run optimization engine
    m.optimize()

    # m.write('TEST3.lp')

    # #return MCP
    # final_mcp = []
    # for v in m.getVars():
    #     if 'M' in v.varName:
    #         final_mcp.append(v.x)

    #return MCP - remove this
    final_mcp = []
    for v in m.getVars():
        final_mcp.append([v.varName,v.x])

    #return obj val
    if obj == 1:
        obj = m.objVal
    elif obj == 2:
        obj = all_demand_area - m.objVal
    elif obj == 3:
        x_vals = { k : v.X for k,v in x.items() }
        y_vals = { k : v.X for k,v in y.items() }
        v_vals = { k : v.X for k,v in vv.items() }
        f_vals = { k : v.X for k,v in f.items() }

        accepted_block_supply_bids = []
        for b in block_supply_bids:
            avg_f = sum([f_vals[h-1] for h in hours_covered[b]])/N[b]
            if block_price[b] < avg_f:
                accepted_block_supply_bids.append(b)

        accepted_block_demand_bids = []
        for b in block_demand_bids:
            avg_f = sum([f_vals[h-1] for h in hours_covered[b]])/N[b]
            if block_price[b] > avg_f:
                accepted_block_demand_bids.append(b)

        accepted_flex_supply_bids = []
        for b in flex_supply_bids:
            max_f = max(f_vals.items())[1]
            if flex_price[b] < max_f:
                accepted_flex_supply_bids.append(b)

        accepted_flex_demand_bids = []
        for b in flex_demand_bids:
            min_f = min(f_vals.items())[1]
            if flex_price[b] > min_f:
                accepted_flex_demand_bids.append(b)

        hourly_demand_area = sum(total_area_demand[(i[0],i[1])] for i in hourly_demand_bids) - sum(0.5*(2*first_price[(i[0],i[1],l)] +  (last_price[(i[0],i[1],l)]-first_price[(i[0],i[1],l)])*x_vals[i[0],i[1],l]) * abs(last_quantity[(i[0],i[1],l)]-first_quantity[(i[0],i[1],l)])*x_vals[i[0],i[1],l] for i in hourly_demand_bids for l in L_i)
        hourly_supply_area = sum(0.5*(2*first_price[(i[0],i[1],l)] +  (last_price[(i[0],i[1],l)]-first_price[(i[0],i[1],l)])*x_vals[i[0],i[1],l]) * (last_quantity[(i[0],i[1],l)]-first_quantity[(i[0],i[1],l)])*x_vals[i[0],i[1],l] for i in hourly_supply_bids for l in L_i)
        # block_supply_obj = sum(N[b]*block_quantity[b]*block_price[b]   for b in accepted_block_supply_bids)
        # block_demand_obj = sum(N[b]*block_quantity[b]*block_price[b]   for b in accepted_block_demand_bids)
        # flex_demand_obj =  sum( flex_quantity[e]*flex_price[e]  for e in accepted_flex_demand_bids)
        # flex_supply_obj = sum( flex_quantity[e]*flex_price[e] for e in accepted_flex_supply_bids)


        block_supply_obj = sum(N[b]*block_quantity[b]*block_price[b]*y_vals[block_bids.index(b)]   for b in block_supply_bids)
        block_demand_obj = sum(N[b]*block_quantity[b]*block_price[b]*y_vals[block_bids.index(b)]   for b in block_demand_bids)
        flex_demand_obj =  sum( flex_quantity[e]*flex_price[e] * sum(v_vals[e,z] for z in Z)  for e in flex_demand_bids)
        flex_supply_obj = sum( flex_quantity[e]*flex_price[e] * sum(v_vals[e,z] for z in Z) for e in flex_supply_bids)

        obj = hourly_demand_area - hourly_supply_area + block_demand_obj - block_supply_obj + flex_demand_obj - flex_supply_obj
    return obj, final_mcp
