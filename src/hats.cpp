#include "hats.hpp"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdio>

typedef struct {
    uint32_t event_count;             // number of events processed in this cell
    std::vector<double> histogram;    // histogram bins (time surface accumulations)
    std::vector<Event_t*> memory;     // memory buffer of past events in this cell (for Δt window)
} Cell_t;

typedef struct {
    uint32_t width;       // sensor width
    uint32_t height;      // sensor height
    uint32_t area;        // total pixels = width × height
    uint32_t cells_width; // number of cells across width
    uint32_t cells_height;// number of cells across height
    uint32_t cells_area;  // total number of cells

    int32_t rho;          // spatial neighborhood radius
    double delta;         // temporal window size delta t
    double tau;           // decay constant τ

    uint32_t cell_dim;    // cell size K

    std::vector<Cell_t*> cells;       // array of cell pointers
    std::vector<Event_t*> events;     // event buffer to process
    std::vector<double> normalized_hist; // final concatenated HATS descriptor
} HATS_Context_t;

// Initialize context with sensor and algorithm parameters
void context_init(HATS_Context_t *ctx, uint32_t width, uint32_t height,
                  uint32_t cell_dim, int rho, double delta, double tau)
{
    ctx->width = width;
    ctx->height = height;
    ctx->area = width * height;
    ctx->cell_dim = cell_dim;

    // compute grid of cells
    ctx->cells_width  = (width  + cell_dim - 1)/cell_dim;
    ctx->cells_height = (height + cell_dim - 1)/cell_dim;
    ctx->cells_area   = ctx->cells_width * ctx->cells_height;

    // initialize cells with nullptrs
    ctx->cells = std::vector<Cell_t*>(ctx->cells_area, nullptr);
    ctx->normalized_hist.clear();

    ctx->rho = rho;
    ctx->delta = delta;
    ctx->tau = tau;
}

void context_free(HATS_Context_t *ctx) {
    for (Cell_t* cell_ptr : ctx->cells) {
        delete cell_ptr;
    }
    ctx->cells.clear();
    ctx->events.clear();
    ctx->normalized_hist.clear();
}

void compute_time_surface(HATS_Context_t *ctx, Cell_t *current_cell, Event_t *event)
{
    uint32_t x = event->x;
    uint32_t y = event->y;
    double t   = event->t;
    uint32_t pol = event->pol;

    // Iterate over past events stored in this cell
    for (Event_t *mem_event : current_cell->memory) {
        uint32_t ex   = mem_event->x;
        uint32_t ey   = mem_event->y;
        double et     = mem_event->t;
        uint32_t epol = mem_event->pol;

        if ((pol != epol) || ((t - et) > ctx->delta)) continue;

        int dx = (int) ex - (int) x;
        int dy = (int) ey - (int) y;

        // Only neighbors within +-rho are used
        if (std::abs(dx) > ctx->rho || std::abs(dy) > ctx->rho) continue;
        
        int32_t x_idx = dx + ctx->rho;
        int32_t y_idx = dy + ctx->rho;
        int32_t p_idx = (pol == 1 ? 1 : 0);

        int32_t s    = 2*ctx->rho + 1;   // neighborhood side length
        int32_t s_sq = s*s;              // neighborhood area

        int32_t histogram_index = p_idx * s_sq + (y_idx * s + x_idx);

        if (histogram_index < 0 || (size_t) histogram_index >= current_cell->histogram.size()) continue;
        
        // Exponential decay weighting
        double weight = std::exp(-1 * ((t - et)/ctx->tau));
        current_cell->histogram[histogram_index] += weight;
    }
}

void cell_init(HATS_Context_t *ctx, Cell_t *cell) 
{
    cell->event_count = 0;
    cell->memory.clear();

    int histogram_size = 2 * (2*ctx->rho + 1) * (2*ctx->rho + 1);
    cell->histogram.assign(histogram_size, 0.0);
}

// Normalize histograms per cell
void normalize_histograms(HATS_Context_t *ctx) 
{
    ctx->normalized_hist.clear();
    for (Cell_t *cell : ctx->cells) {
        if (cell == nullptr) {
            int hist_size = 2 * (2*ctx->rho + 1) * (2*ctx->rho + 1);
            for (int i = 0; i < hist_size; i++) {
                ctx->normalized_hist.push_back(0.0);
            }
            continue;
        }

        if (cell->event_count == 0) {
            for (size_t i = 0; i < cell->histogram.size(); i++) {
                ctx->normalized_hist.push_back(0.0);
            }
            continue;
        }
        
        for (const double& h_val : cell->histogram) {
            double norm_val = h_val/cell->event_count;
            ctx->normalized_hist.push_back(norm_val);
        }
    }
}

// keep events within delta t of current time
void prune_memory(Cell_t* cell, double t_now, double delta) {
    size_t write = 0;
    for (size_t read = 0; read < cell->memory.size(); ++read) {
        if (cell->memory[read]->t >= t_now - delta) {
            cell->memory[write++] = cell->memory[read];
        }
    }
    cell->memory.resize(write);
}

// Main HATS pipeline: iterate over events, update cells, build features
void process(HATS_Context_t *ctx)
{
    for (Event_t *event : ctx->events) {

        size_t x_id = event->x / ctx->cell_dim;
        size_t y_id = event->y / ctx->cell_dim;
        size_t idx_id = y_id * ctx->cells_width + x_id;

        if (idx_id >= ctx->cells.size()) {
            printf("idx out of bounds in process \n");
            continue;
        }

        if (ctx->cells[idx_id] == nullptr) {
            ctx->cells[idx_id] = new Cell_t();
            cell_init(ctx, ctx->cells[idx_id]);
        }

        Cell_t *current_cell = ctx->cells[idx_id];
        compute_time_surface(ctx, current_cell, event);

        current_cell->memory.push_back(event);
        current_cell->event_count += 1;

        prune_memory(current_cell, event->t, ctx->delta);
    }

    normalize_histograms(ctx);
}

// wrapper: process a batch of events into a feature vector
std::vector<double> hats_processing(Event_t *input, int32_t size, int32_t width, int32_t height, int32_t cell_dim, int32_t rho, double delta, double tau) 
{
    HATS_Context_t ctx;
    context_init(&ctx, width, height, cell_dim, rho, delta, tau);
    for (int i = 0; i < size; i++) {
        ctx.events.push_back(&input[i]);
    }
    process(&ctx);
    std::vector<double> result_vector(ctx.normalized_hist);
    context_free(&ctx);
    return result_vector;
}