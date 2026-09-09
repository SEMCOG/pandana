#include "accessibility.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>
#include "graphalg.h"

namespace MTC {
namespace accessibility {

using std::string;
using std::vector;
using std::pair;
using std::make_pair;

typedef std::pair<double, int> distance_node_pair;
bool distance_node_pair_comparator(const distance_node_pair& l,
                                   const distance_node_pair& r)
    { return l.first < r.first; }


Accessibility::Accessibility(
        int numnodes,
        vector< vector<PandanaNodeID>> edges,
        vector< vector<double>>  edgeweights,
        bool twoway) {

    this->aggregations.reserve(9);
    this->aggregations.push_back("sum");
    this->aggregations.push_back("mean");
    this->aggregations.push_back("min");
    this->aggregations.push_back("25pct");
    this->aggregations.push_back("median");
    this->aggregations.push_back("75pct");
    this->aggregations.push_back("max");
    this->aggregations.push_back("std");
    this->aggregations.push_back("count");

    this->decays.reserve(3);
    this->decays.push_back("exp");
    this->decays.push_back("linear");
    this->decays.push_back("flat");
    this->decays.push_back("friction_curve");

    for (int i = 0 ; i < edgeweights.size() ; i++) {
        this->addGraphalg(new Graphalg(numnodes, edges, edgeweights[i],
                          twoway));
    }

    this->numnodes = numnodes;
    this->dmsradius = -1;
}


void Accessibility::addGraphalg(MTC::accessibility::Graphalg *g) {
    std::shared_ptr<MTC::accessibility::Graphalg>ptr(g);
    this->ga.push_back(ptr);
}


void
Accessibility::precomputeRangeQueries(float radius) {
    dms.resize(ga.size());
    for (int i = 0 ; i < ga.size() ; i++) {
        dms[i].resize(numnodes);
    }

    #pragma omp parallel
    {
    #pragma omp for schedule(guided)
    for (int i = 0 ; i < numnodes ; i++) {
        for (int j = 0 ; j < ga.size() ; j++) {
            ga[j]->Range(
                i,
                radius,
                omp_get_thread_num(),
                dms[j][i]);
        }
    }
    }
    dmsradius = radius;
}


vector<vector<pair<PandanaNodeID, float>>>
Accessibility::Range(vector<PandanaNodeID> srcnodes, float radius, int graphno,
                     vector<PandanaNodeID> ext_ids) {

    // Set up a mapping between the external node ids and internal ones
    std::unordered_map<PandanaNodeID, int> int_ids(ext_ids.size());
    for (int i = 0; i < ext_ids.size(); i++) {
        int_ids.insert(pair<PandanaNodeID, int>(ext_ids[i], i));
    }
    
    // use cached results if available
    vector<DistanceVec> dists(srcnodes.size());
    if (dmsradius > 0 && radius <= dmsradius) {
        for (int i = 0; i < srcnodes.size(); i++) {
            dists[i] = dms[graphno][int_ids[srcnodes[i]]];
        }
    }
    else {
        #pragma omp parallel
        #pragma omp for schedule(guided)
        for (int i = 0; i < srcnodes.size(); i++) {
            ga[graphno]->Range(int_ids[srcnodes[i]], radius,
                omp_get_thread_num(), dists[i]);
        }
    }
    
    // todo: check that results are returned from cache correctly
    // todo: check that performing an aggregation creates cache

    // Convert back to external node ids
    vector<vector<pair<PandanaNodeID, float>>> output(dists.size());
    for (int i = 0; i < dists.size(); i++) {
        output[i].resize(dists[i].size());
        for (int j = 0; j < dists[i].size(); j++) {
            output[i][j] = std::make_pair(ext_ids[dists[i][j].first], 
                                          dists[i][j].second);
        }
    }
    return output;
}


vector<int>
Accessibility::Route(int src, int tgt, int graphno) {
    vector<NodeID> ret = this->ga[graphno]->Route(src, tgt);
    return vector<int> (ret.begin(), ret.end());
}


vector<vector<int>>
Accessibility::Routes(vector<PandanaNodeID> sources, vector<PandanaNodeID> targets, int graphno) {

    int n = std::min(sources.size(), targets.size()); // in case lists don't match
    vector<vector<int>> routes(n);

    #pragma omp parallel
    #pragma omp for schedule(guided)
    for (int i = 0 ; i < n ; i++) {
        vector<NodeID> ret = this->ga[graphno]->Route(sources[i], targets[i], 
            omp_get_thread_num());
        routes[i] = vector<int> (ret.begin(), ret.end());
    }
    return routes;
}


double
Accessibility::Distance(int src, int tgt, int graphno) {
    return this->ga[graphno]->Distance(src, tgt);
}


vector<double>
Accessibility::Distances(
        vector<PandanaNodeID> sources,
        vector<PandanaNodeID> targets,
        int graphno) {
    
    int n = std::min(sources.size(), targets.size()); // in case lists don't match
    vector<double> distances(n);
    
    #pragma omp parallel
    #pragma omp for schedule(guided)
    for (int i = 0 ; i < n ; i++) {
        distances[i] = this->ga[graphno]->Distance(
            sources[i], 
            targets[i], 
            omp_get_thread_num());
    }
    return distances;
}


/*
#######################
POI QUERIES
#######################
*/


void Accessibility::initializeCategory(const double maxdist, const int maxitems,
                                       string category, vector<PandanaNodeID> node_idx)
{
    accessibility_vars_t av;
    av.resize(this->numnodes);

    this->maxdist = maxdist;
    this->maxitems = maxitems;

    // initialize for all subgraphs
    for (int i = 0 ; i < ga.size() ; i++) {
        ga[i]->initPOIIndex(category, this->maxdist, this->maxitems);
        // initialize for each node
        for (int j = 0 ; j < node_idx.size() ; j++) {
            int node_id = static_cast<int>(node_idx[j]);

            ga[i]->addPOIToIndex(category, node_id);
            assert(node_id < av.size());
            av[node_id].push_back(j);
        }
    }
    accessibilityVarsForPOIs[category] = av;
}


/* the return_nodeidx parameter determines whether to
   return the nodeidx where the poi was found rather than
   the distances - you can call this twice - once for the
   distances and then again for the node idx */
vector<pair<double, int>>
Accessibility::findNearestPOIs(int srcnode, float maxradius, unsigned number,
                               string cat, int gno)
{
    DistanceMap distancesmap = ga[gno]->NearestPOI(cat, srcnode,
        maxradius, number, omp_get_thread_num());

    vector<distance_node_pair> distance_node_pairs;
    std::map<POIKeyType, accessibility_vars_t>::iterator cat_for_pois = 
        accessibilityVarsForPOIs.find(cat);
    if(cat_for_pois == accessibilityVarsForPOIs.end())
        return distance_node_pairs;
    
    accessibility_vars_t &vars = cat_for_pois->second;

    /* need to account for the possibility of having
     multiple locations at single node */
    for (DistanceMap::const_iterator itDist = distancesmap.begin();
       itDist != distancesmap.end();
       ++itDist) {
      int nodeid = itDist->first;
      double distance = itDist->second;

      for (int i = 0 ; i < vars[nodeid].size() ; i++) {
          distance_node_pairs.push_back(
             make_pair(distance, static_cast<int>(vars[nodeid][i])));
      }
    }

    std::sort(distance_node_pairs.begin(), distance_node_pairs.end(),
            distance_node_pair_comparator);

    return distance_node_pairs;
}


/* the return_nodeds param is described above */
pair<vector<vector<double>>, vector<vector<int>>>
Accessibility::findAllNearestPOIs(float maxradius, unsigned num_of_pois,
                                  string category,int gno)
{
    vector<vector<double>>
        dists(numnodes, vector<double> (num_of_pois));

    vector<vector<int>>
        poi_ids(numnodes, vector<int> (num_of_pois));

    #pragma omp parallel for
    for (int i = 0 ; i < numnodes ; i++) {
        vector<pair<double, int>> d = findNearestPOIs(
            i,
            maxradius,
            num_of_pois,
            category,
            gno);
        for (int j = 0 ; j < num_of_pois ; j++) {
            if (j < d.size()) {
                dists[i][j] = d[j].first;
                poi_ids[i][j] = d[j].second;
            } else {
                dists[i][j] = -1;
                poi_ids[i][j] = -1;
            }
        }
    }
    return make_pair(dists, poi_ids);
}


/*
#######################
AGGREGATION/ACCESSIBILITY QUERIES
#######################
*/


void Accessibility::initializeAccVar(
    string category,
    vector<PandanaNodeID> node_idx,
    vector<double> values) {
    accessibility_vars_t av;
    av.resize(this->numnodes);
    for (int i = 0 ; i < node_idx.size() ; i++) {
        int node_id = static_cast<int>(node_idx[i]);
        double val = values[i];

        assert(node_id < av.size());
        av[node_id].push_back(val);
    }
    accessibilityVars[category] = av;
}


vector<double>
Accessibility::getAllAggregateAccessibilityVariables(
    float radius,
    string category,
    string aggtyp,
    string decay,
    int graphno) {
    if (accessibilityVars.find(category) == accessibilityVars.end() ||
        std::find(aggregations.begin(), aggregations.end(), aggtyp)
            == aggregations.end() ||
        std::find(decays.begin(), decays.end(), decay) == decays.end()) {
        // not found
        return vector<double>();
    }

    vector<double> scores(numnodes);

    #pragma omp parallel
    {
    #pragma omp for schedule(guided)
    for (int i = 0 ; i < numnodes ; i++) {
        scores[i] = aggregateAccessibilityVariable(
            i,
            radius,
            accessibilityVars[category],
            aggtyp,
            decay,
            graphno);
    }
    }
    return scores;
}


double
Accessibility::quantileAccessibilityVariable(
    const DistanceVec &distances,
    accessibility_vars_t &vars,
    float quantile,
    float radius) {

    // first iterate through nodes in order to get count of items
    int cnt = 0;

    // distances is sorted ascending (a guaranteed property of the underlying
    // Dijkstra-style range search, see Graphalg::Range), so once one entry
    // exceeds radius every remaining entry does too -- break rather than
    // continuing to check (and skip) the rest of a potentially much larger
    // cached list.
    for (int i = 0 ; i < distances.size() ; i++) {
        int nodeid = distances[i].first;
        double distance = distances[i].second;

        if (distance > radius) break;

        cnt += vars[nodeid].size();
    }

    if (cnt == 0) return -1;

    vector<float> vals(cnt);

    // make a second pass to put items in a single array for sorting
    for (int i = 0, cnt = 0 ; i < distances.size() ; i++) {
        int nodeid = distances[i].first;
        double distance = distances[i].second;

        if (distance > radius) break;

        // and then iterate through all items at the node
        for (int j = 0 ; j < vars[nodeid].size() ; j++)
            vals[cnt++] = vars[nodeid][j];
    }

    std::sort(vals.begin(), vals.end());

    int ind = static_cast<int>(vals.size() * quantile);

    if (quantile <= 0.0) ind = 0;
    if (quantile >= 1.0) ind = vals.size()-1;

    return vals[ind];
}


double
Accessibility::aggregateAccessibilityVariable(
    int srcnode,
    float radius,
    accessibility_vars_t &vars,
    string aggtyp,
    string decay,
    int gno) {
    // Bind to the cached range-query result via pointer instead of copying
    // it. A C++ reference can't be re-pointed after initialization, so the
    // previous "DistanceVec &distances = tmp; distances = dms[...];" pattern
    // was actually a full vector copy through the reference on every call,
    // despite the original intent (avoiding a copy in the precompute case).
    DistanceVec tmp;
    const DistanceVec *distances_ptr;
    if (dmsradius > 0 && radius <= dmsradius) {
        distances_ptr = &dms[gno][srcnode];
    } else {
        ga[gno]->Range(
            srcnode,
            radius,
            omp_get_thread_num(),
            tmp);
        distances_ptr = &tmp;
    }
    const DistanceVec &distances = *distances_ptr;

    if (distances.size() == 0) return -1;

    if (aggtyp == "max" || aggtyp == "min") {
        // Streaming extremum instead of quantileAccessibilityVariable's
        // sort-based path below: min/max only need a single running-best
        // pass (O(k), no extra allocation), not a full sort (O(k log k)) of
        // every candidate value just to read off the first or last entry.
        // Compared as float, matching accessibility_vars_t's actual storage
        // type (vector<vector<float>>), so this returns the same value the
        // old sort-based path would have.
        bool is_max = (aggtyp == "max");
        float best = is_max ? -std::numeric_limits<float>::infinity()
                             :  std::numeric_limits<float>::infinity();
        bool found = false;
        // distances is sorted ascending (see Graphalg::Range), so this can
        // stop at the first out-of-radius entry instead of scanning the rest.
        for (int i = 0 ; i < distances.size() ; i++) {
            int nodeid = distances[i].first;
            double distance = distances[i].second;

            if (distance > radius) break;

            for (int j = 0 ; j < vars[nodeid].size() ; j++) {
                found = true;
                float v = vars[nodeid][j];
                if (is_max ? (v > best) : (v < best)) best = v;
            }
        }
        return found ? best : -1;
    } else if (aggtyp == "25pct") {
        return this->quantileAccessibilityVariable(
            distances, vars, 0.25, radius);
    } else if (aggtyp == "median") {
        return this->quantileAccessibilityVariable(
            distances, vars, 0.5, radius);
    } else if (aggtyp == "75pct") {
        return this->quantileAccessibilityVariable(
            distances, vars, 0.75, radius);
    }

    if (aggtyp == "std") decay = "flat";

    int cnt = 0;
    double sum = 0.0;
    double sumsq = 0.0;

    std::function<double(const double &, const float &, const float &)> sum_function;

    if(decay == "exp")
        sum_function = [](const double &distance, const float &radius, const float &var)
                        { return exp(-1*distance/radius) * var; };
    if(decay == "linear")
        sum_function = [](const double &distance, const float &radius, const float &var)
                        { return (1.0-distance/radius) * var; };
    if(decay == "flat")
        sum_function = [](const double &distance, const float &radius, const float &var)
                        { return var; };
    if (decay == "friction_curve") {
        // Gamma + offset
        // f(d) = c + alpha * d^(k-1) * exp(- d / theta)
        // best fit:
        //   c=177.0997, alpha=28146.5031, k=2.4613, theta=1.77
        double c      = 177.0997;
        double alpha  = 28146.5031;
        double k      = 2.4613;
        double theta  = 1.77;

        sum_function = [c, alpha, k, theta](const double &distance, const float &radius, const float &val)
        {
            // If distance <= 0, watch out for pow(d^(k-1))
            // assume distances > 0. If some distance is 0, add a small epsilon.
            double d_eff = std::max(distance, 1e-4);

            double gamma_val = c + alpha * std::pow(d_eff, k - 1.0) * std::exp(-d_eff / theta);

            // Multiply by val. This is your 'friction' times the variable from "vars"
            return gamma_val * val;
        };
    }

    // Sum across all nodes within radius. distances is sorted ascending
    // (see Graphalg::Range), so once one entry exceeds radius every
    // remaining entry does too -- stop here rather than walking the rest of
    // a cached list that may be sized for a much larger radius.
    for (int i = 0 ; i < distances.size() ; i++) {
        int nodeid = distances[i].first;
        double distance = distances[i].second;

        if (distance > radius) break;

        for (int j = 0 ; j < vars[nodeid].size() ; j++) {
            cnt++;  // count items
            sum += sum_function(distance, radius, vars[nodeid][j]);

            // stddev is always flat
            sumsq += vars[nodeid][j] * vars[nodeid][j];
        }
    }

    if (aggtyp == "count") return cnt;

    if (aggtyp == "mean" && cnt != 0) sum /= cnt;

    if (aggtyp == "std" && cnt != 0) {
        double mean = sum / cnt;
        return sqrt(sumsq / cnt - mean * mean);
    }

    return sum;
}

}  // namespace accessibility
}  // namespace MTC
