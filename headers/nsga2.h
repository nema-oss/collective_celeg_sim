//
// Created by nema on 10/11/26.
//

#ifndef UNTITLED2_NSGA2_H
#define UNTITLED2_NSGA2_H
#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include "run.h"
#include <sstream>

namespace nsga {

    using json = nlohmann::json;

    struct Gene { double lo, hi; };

    struct Layout {
        std::vector<Gene> genes;
        int l2[9][3];
        int w_phero, w_food;

        Layout() {
            const Gene coeff{-5.0f, 5.0f}, intercept{-2.5, 2.5};
            genes = {coeff, intercept, {0.0, 1.0}};
            for (int t = 0; t < 9; ++t) {
                l2[t][0] = genes.size(); genes.push_back(coeff);
                l2[t][1] = genes.size(); genes.push_back(intercept);
                if (t / 3 == t % 3) {
                    l2[t][2] = -1;
                } else {
                    l2[t][2] = genes.size(); genes.push_back({0.0, 2.0});
                }
            }
            w_phero = genes.size(); genes.push_back({W_MIN, W_MAX});
            w_food  = genes.size(); genes.push_back({W_MIN, W_MAX});
        }

        std::vector<double> decode(const std::vector<double>& x) const {
            std::vector<double> v(x.size());
            for (size_t i = 0; i < x.size(); ++i)
                v[i] = genes[i].lo + x[i] * (genes[i].hi - genes[i].lo);
            return v;
        }

        std::vector<double> encode(const std::vector<double>& v) const {
            std::vector<double> x(v.size());
            for (size_t i = 0; i < v.size(); ++i)
                x[i] = std::min(1.0, std::max(0.0, (v[i] - genes[i].lo) / (genes[i].hi - genes[i].lo)));
            return x;
        }

        std::string header() const {
            std::string h = "gen,rank,nu,fc,frac,l1_coeff,l1_intercept,l1_height";
            for (int t = 0; t < 9; ++t) {
                const std::string p = ",l2_" + std::to_string(t / 3) + std::to_string(t % 3);
                h += p + "_coeff" + p + "_intercept";
                if (l2[t][2] >= 0) h += p + "_height";
            }
            return h + ",w_phero,w_food\n";
        }
    };

    struct Ind {
        std::vector<double> x;
        double f[2]  = {0, 0};
        double nu    = 0;
        double fc    = 0;
        int    rank  = 0;
        double crowd = 0;
        double frac  = 0;     // fraction of food consumed, logged only
    };

    static json entry(double coeff, double intercept, double height, double p_off) {
        return {{"p_off_food", p_off}, {"tau", -1}, {"model_coeff", coeff},
                {"model_intercept", intercept}, {"model_height", height},
                {"mean", 0}, {"std", 1}, {"p_relevant", 0}, {"sign", 1}};
    }

    static void set_objectives(Ind& ind) {
        ind.f[0] = DISPERSE ? ind.nu : -ind.nu;
        ind.f[1] = -ind.fc;
    }

    struct Problem {
        Layout  layout;
        json    off;
        Options opt;

        explicit Problem(const Options& base) : opt(base) {
            opt.quiet   = true;
            opt.l1_path = "/state_estimations/l1.json";
            opt.l2_path = "/state_estimations/l2.json";
            std::ifstream in("/state_estimations/off_food_transitions.json");
            if (!in) throw std::runtime_error("cannot open off_food_transitions.json");
            off = json::parse(in);
        }

        void write_candidate(const std::vector<double>& v) const {
            json l1, l2;
            l1["2"] = entry(v[0], v[1], v[2], off["2"]["2"].get<double>());
            for (int t = 0; t < 9; ++t) {
                const std::string src = std::to_string(t / 3), dst = std::to_string(t % 3);
                const int h = layout.l2[t][2];
                l2[src][dst] = entry(v[layout.l2[t][0]], v[layout.l2[t][1]],
                                     h < 0 ? 1.0 : v[h], off[src][dst].get<double>());
            }
            std::ofstream(opt.l1_path) << l1.dump(2);
            std::ofstream(opt.l2_path) << l2.dump(2);
        }

        void evaluate(Ind& ind) {
            const std::vector<double> v = layout.decode(ind.x);
            write_candidate(v);

            Options o = opt;
            o.w_phero = v[layout.w_phero];
            o.w_food  = v[layout.w_food];

            double nu_sum = 0, fc_sum = 0, frac_sum = 0;
            bool ok = true;
            for (int s = 0; s < opt.eval_seeds && ok; ++s) {
                o.seed = opt.seed + s;
                double nu = 0, frac = 0, intake = 0;
                ok = run_once(o, nu, &frac, &intake);
                nu_sum   += nu;
                fc_sum   += intake;
                frac_sum += frac;
            }

            if (!ok) {
                ind.nu = ind.fc = ind.frac = -1e9;
            } else {
                ind.nu   = nu_sum / opt.eval_seeds;
                ind.fc   = fc_sum / opt.eval_seeds;
                ind.frac = frac_sum / opt.eval_seeds;
            }
            set_objectives(ind);
        }
    };

    static bool dominates(const Ind& a, const Ind& b) {
        return a.f[0] <= b.f[0] && a.f[1] <= b.f[1] && (a.f[0] < b.f[0] || a.f[1] < b.f[1]);
    }

    static std::vector<std::vector<int>> sort_fronts(std::vector<Ind>& pop) {
        const int n = pop.size();
        std::vector<std::vector<int>> dominated(n), fronts(1);
        std::vector<int> count(n, 0);

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (dominates(pop[i], pop[j])) dominated[i].push_back(j);
                else if (dominates(pop[j], pop[i])) ++count[i];
            }

        for (int i = 0; i < n; ++i)
            if (count[i] == 0) { pop[i].rank = 0; fronts[0].push_back(i); }

        for (size_t k = 0; !fronts[k].empty(); ++k) {
            std::vector<int> next;
            for (int i : fronts[k])
                for (int j : dominated[i])
                    if (--count[j] == 0) { pop[j].rank = k + 1; next.push_back(j); }
            fronts.push_back(next);
        }
        fronts.pop_back();
        return fronts;
    }

    static void crowding(std::vector<Ind>& pop, const std::vector<int>& front) {
        for (int i : front) pop[i].crowd = 0;
        for (int m = 0; m < 2; ++m) {
            std::vector<int> idx = front;
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return pop[a].f[m] < pop[b].f[m]; });
            const double range = pop[idx.back()].f[m] - pop[idx.front()].f[m];
            pop[idx.front()].crowd = pop[idx.back()].crowd = INFINITY;
            if (range <= 0) continue;
            for (size_t k = 1; k + 1 < idx.size(); ++k)
                pop[idx[k]].crowd += (pop[idx[k + 1]].f[m] - pop[idx[k - 1]].f[m]) / range;
        }
    }

    static bool better(const Ind& a, const Ind& b) {
        return a.rank < b.rank || (a.rank == b.rank && a.crowd > b.crowd);
    }

    static const Ind& tournament(const std::vector<Ind>& pop, std::mt19937& rng) {
        std::uniform_int_distribution<int> pick(0, pop.size() - 1);
        const Ind& a = pop[pick(rng)];
        const Ind& b = pop[pick(rng)];
        return better(a, b) ? a : b;
    }

    static Ind make_child(const Ind& p1, const Ind& p2, std::mt19937& rng) {
        constexpr double ETA_C = 15.0, ETA_M = 20.0;
        std::uniform_real_distribution<double> U(0.0, 1.0);
        const size_t n = p1.x.size();

        Ind c;
        c.x.resize(n);
        for (size_t i = 0; i < n; ++i) {
            double x = p1.x[i];
            if (U(rng) < 0.5) {
                const double u = U(rng);
                const double beta = u <= 0.5 ? std::pow(2 * u, 1.0 / (ETA_C + 1))
                                             : std::pow(1.0 / (2 * (1 - u)), 1.0 / (ETA_C + 1));
                x = 0.5 * ((1 + beta) * p1.x[i] + (1 - beta) * p2.x[i]);
            }
            if (U(rng) < 1.0 / n) {
                const double u = U(rng);
                x += u < 0.5 ? std::pow(2 * u, 1.0 / (ETA_M + 1)) - 1
                             : 1 - std::pow(2 * (1 - u), 1.0 / (ETA_M + 1));
            }
            c.x[i] = std::min(1.0, std::max(0.0, x));
        }
        return c;
    }

    static void log_population(std::ofstream& csv, const Problem& pb, const std::vector<Ind>& pop, int gen) {
        for (const Ind& ind : pop) {
            csv << gen << ',' << ind.rank << ',' << ind.nu << ',' << ind.fc << ',' << ind.frac;
            for (double v : pb.layout.decode(ind.x)) csv << ',' << v;
            csv << '\n';
        }
        csv.flush();
    }

    using Clock = std::chrono::steady_clock;

    static std::string fmt_hms(double seconds) {
        long s = (long)seconds;
        char buf[32];
        snprintf(buf, sizeof(buf), "%02ld:%02ld:%02ld", s / 3600, (s % 3600) / 60, s % 60);
        return buf;
    }

    struct Progress {
        int    total;
        int    done = 0;
        double busy = 0.0;      // seconds spent inside evaluations
        double first = 0.0;     // first eval, includes CUDA warm-up
        Clock::time_point start = Clock::now();

        explicit Progress(int total_evals) : total(total_evals) {}

        template <class F>
        void timed(F&& eval) {
            const auto t0 = Clock::now();
            eval();
            const double dt = std::chrono::duration<double>(Clock::now() - t0).count();

            if (done == 0) first = dt;
            else busy += dt;
            ++done;

            // steady-state mean excludes the warm-up eval once we have another sample
            const double mean = done > 1 ? busy / (done - 1) : first;
            const double eta  = mean * (total - done);
            printf("  eval %d/%d  last %.1fs  mean %.1fs  elapsed %s  eta %s\n",
                   done, total, dt, mean,
                   fmt_hms(std::chrono::duration<double>(Clock::now() - start).count()).c_str(),
                   fmt_hms(eta).c_str());
        }
    };

    struct Stall {
        double best_nu = -INFINITY, best_fc = -INFINITY;
        int    stale   = 0;

        bool update(const std::vector<Ind>& pop, double tol_nu, double tol_fc, int patience) {
            double nu = -INFINITY, fc = -INFINITY;
            for (const Ind& ind : pop) {
                nu = std::max(nu, ind.nu);
                fc = std::max(fc, ind.fc);
            }
            const bool improved = nu > best_nu + tol_nu || fc > best_fc + tol_fc;
            best_nu = std::max(best_nu, nu);
            best_fc = std::max(best_fc, fc);
            stale = improved ? 0 : stale + 1;
            return patience > 0 && stale >= patience;
        }
    };

    static bool load_history(const std::string& path, const Layout& layout, int pop_size,
                             std::vector<Ind>& pop, int& last_gen) {
        std::ifstream in(path);
        if (!in) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }

        const size_t n_cols = 5 + layout.genes.size();   // gen,rank,nu,fc,frac + genes
        std::string header, line;
        std::getline(in, header);
        if (header != layout.header().substr(0, layout.header().size() - 1)) {
            fprintf(stderr, "history header does not match the current gene layout\n");
            return false;
        }

        struct Row { int gen; std::vector<double> v; std::string text; };
        std::vector<Row> rows;
        std::map<int, int> per_gen;

        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> v;
            try {
                while (std::getline(ss, cell, ',')) v.push_back(std::stod(cell));
            } catch (...) { break; }                       // truncated last line
            if (v.size() != n_cols) break;
            rows.push_back({(int)v[0], v, line});
            ++per_gen[(int)v[0]];
        }

        last_gen = -1;
        for (const auto& kv : per_gen)
            if (kv.second == pop_size) last_gen = kv.first;
        if (last_gen < 0) { fprintf(stderr, "no complete generation in %s\n", path.c_str()); return false; }

        pop.clear();
        std::ofstream out(path, std::ios::trunc);          // drop any partial generation after last_gen
        out << header << '\n';
        for (const Row& r : rows) {
            if (r.gen > last_gen) continue;
            out << r.text << '\n';
            if (r.gen != last_gen) continue;

            Ind ind;
            ind.nu   = r.v[2];
            ind.fc   = r.v[3];
            ind.frac = r.v[4];
            ind.f[0] = -ind.nu;
            ind.f[1] = -ind.fc;
            ind.x    = layout.encode(std::vector<double>(r.v.begin() + 5, r.v.end()));
            pop.push_back(std::move(ind));
        }
        return true;
    }

    struct Scale { double nu, fc; };

    static double hypervolume(const std::vector<Ind>& pop, const Scale& s) {
        std::vector<std::pair<double, double>> pts;          // normalized (nu, fc)
        for (const Ind& ind : pop)
            if (ind.rank == 0 && ind.nu >= 0 && ind.fc >= 0){  // skips failed evals (-1e9)
                const double a = DISPERSE ? (s.nu - ind.nu) : ind.nu;
                pts.push_back({a / s.nu, ind.fc / s.fc});
            }
        std::sort(pts.rbegin(), pts.rend());                  // nu descending

        double hv = 0, run_max = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            run_max = std::max(run_max, pts[i].second);
            const double next_nu = i + 1 < pts.size() ? pts[i + 1].first : 0.0;
            hv += (pts[i].first - next_nu) * run_max;
        }
        return hv;
    }

    struct HvStall {
        double best  = 0;
        int    stale = 0;

        bool update(double hv, double tol, int patience) {
            if (hv > best * (1.0 + tol)) stale = 0; else ++stale;
            best = std::max(best, hv);
            return patience > 0 && stale >= patience;
        }
    };

    static int run(const Options& base) {
        if (base.condition == Condition::Off)
            fprintf(stderr, "warning: condition is off, food consumed is always 0\n");

        Problem pb(base);
        const SimConfig cfg = make_config(base);
        const Scale scale{
                std::max(1.0, base.worm_count - 1.0),                          // max possible neighbors
                std::max(1e-9, (double)GRID_N * GRID_N * cfg.food_amount / base.worm_count)  // intake if all food is eaten
        };
        std::mt19937 rng(base.seed);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        const int n_genes = pb.layout.genes.size();

        make_dirs(base.out_dir + "/nsga");
        const std::string hist = base.out_dir + "/nsga/history.csv";

        std::vector<Ind> pop;
        int start_gen = 1;

        if (base.resume) {
            int last_gen = 0;
            if (!load_history(hist, pb.layout, base.pop, pop, last_gen)) return 1;
            start_gen = last_gen + 1;
            rng.seed(base.seed + 1000 * last_gen);          // fresh stream, not a replay of the original
            printf("resuming from gen %d (%d individuals), target gen %d\n", last_gen, (int)pop.size(), base.gens);
        }

        std::ofstream csv(hist, base.resume ? std::ios::app : std::ios::trunc);
        csv.precision(10);

        const int remaining = std::max(0, base.gens - start_gen + 1);
        Progress progress(base.resume ? base.pop * remaining : base.pop * (base.gens + 1));

        if (!base.resume) {
            csv << pb.layout.header();
            pop.resize(base.pop);
            for (Ind& ind : pop) {
                ind.x.resize(n_genes);
                for (double& g : ind.x) g = U(rng);
                progress.timed([&] { pb.evaluate(ind); });
            }
        }

        for (auto& fr : sort_fronts(pop)) crowding(pop, fr);

        if (!base.resume) {
            log_population(csv, pb, pop, 0);
            const double gen0 = std::chrono::duration<double>(Clock::now() - progress.start).count();
            printf("gen 0 took %s; estimated total ~%s for %d evals\n",
                   fmt_hms(gen0).c_str(), fmt_hms(gen0 / base.pop * progress.total).c_str(), progress.total);
        }

        Stall stall;   // if you added early stopping; it restarts from scratch on resume
        // initial population
        for (Ind& ind : pop) {
            ind.x.resize(n_genes);
            for (double& g : ind.x) g = U(rng);
            progress.timed([&] { pb.evaluate(ind); });
        }
        for (auto& fr : sort_fronts(pop)) crowding(pop, fr);
        log_population(csv, pb, pop, 0);
        const double gen0 = std::chrono::duration<double>(Clock::now() - progress.start).count();
        printf("gen 0 took %s; estimated total ~%s for %d evals\n",
               fmt_hms(gen0).c_str(), fmt_hms(gen0 / base.pop * progress.total).c_str(), progress.total);
        HvStall hv_stall;
        hv_stall.best = hypervolume(pop, scale);
        for (int gen = start_gen; gen <= base.gens; ++gen) {
            std::vector<Ind> all = pop;
            for (int i = 0; i < base.pop; ++i) {
                // children
                Ind child = make_child(tournament(pop, rng), tournament(pop, rng), rng);
                progress.timed([&] { pb.evaluate(child); });
                all.push_back(std::move(child));
            }

            std::vector<Ind> next;
            for (auto& fr : sort_fronts(all)) {
                crowding(all, fr);
                std::vector<int> order = fr;
                if (next.size() + fr.size() > (size_t)base.pop)
                    std::sort(order.begin(), order.end(), [&](int a, int b) { return all[a].crowd > all[b].crowd; });
                for (int i : order) {
                    if (next.size() == (size_t)base.pop) break;
                    next.push_back(all[i]);
                }
                if (next.size() == (size_t)base.pop) break;
            }
            pop = std::move(next);

            int front0 = 0;
            for (const Ind& ind : pop) front0 += ind.rank == 0;
            //printf("gen %d: front size %d\n", gen, front0);
            log_population(csv, pb, pop, gen);
            const double hv = hypervolume(pop, scale);
            printf("gen %d: front size %d, hv %.5f\n", gen, front0, hv);   // replaces the old front-size printf
            if (hv_stall.update(hv, base.tol_hv, base.patience)) {
                printf("stopping at gen %d: hypervolume improved <%.1f%% for %d generations\n",
                       gen, 100 * base.tol_hv, base.patience);
                break;
            }
            /*if (stall.update(pop, base.tol_nu, base.tol_fc, base.patience)) {
                printf("stopping at gen %d: no improvement beyond tolerance for %d generations\n",
                       gen, base.patience);
                break;
            }*/
        }

        printf("pareto front (nu, fc, w_phero, w_food):\n");
        for (const Ind& ind : pop)
            if (ind.rank == 0) {
                const std::vector<double> v = pb.layout.decode(ind.x);
                printf("  %.3f  %.4f  %.3f  %.3f\n", ind.nu, ind.fc,
                       v[pb.layout.w_phero], v[pb.layout.w_food]);
            }
        return 0;
    }

}  // namespace nsga
#endif //UNTITLED2_NSGA2_H
