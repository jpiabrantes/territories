// Core simulation for artificial life with kinship, resources, and territory control

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "raylib.h"
#include "utils/io.h"
#include "utils/helper.h"
#include "utils/bitset.h"

// ========== CORE CONSTANTS ==========

// Game Constants
#define K 0.07167543
#define MAX_GROWTH_DURATION 70
#define STARTING_DAY 55
#define SUMMER_DURATION 100
#define WINTER_DURATION 10
#define WALL_HP_MAX 8
#define MAX_SATIATION 100
#define MAX_HP 3
#define MAX_FOOD_CARRYING_CAPACITY 150
#define MAX_STONE_CARRYING_CAPACITY 10
#define MAX_FOOD_STORAGE_CAPACITY 150
#define STONE_MINED_PER_TICK 5
#define STONE_PER_MINE 600
#define VISION_RADIUS 4
#define METABOLISM_RATE 5
#define REPRODUCTION_AGE 10

// Rendering Constants
#define FRAME_RATE 60
#define TILE_SIZE 64
#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720
#define GAME_WIDTH 1000
#define GAME_HEIGHT 720
#define SPRITE_SIZE 128

// Utility macros
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

// ========== ENUMS ==========

typedef enum {
    MOVE_UP = 0,
    MOVE_RIGHT = 1,
    MOVE_DOWN = 2,
    MOVE_LEFT = 3,
    NOOP = 4,
    PICKUP = 5, // food
    MINE = 6, // stone
    PACKAGE_FOOD = 7, // package food being carried or/and crop on the tile
    BUILD_WALL = 8,
    ATTACK = 9,
    REPRODUCE = 10
} Action;

typedef enum {
    DIR_UP = 0,
    DIR_RIGHT = 1,
    DIR_DOWN = 2,
    DIR_LEFT = 3
} Direction;

typedef enum {
    LAST_HARVEST = 0,
    STORED_FOOD = 1,
    STONE = 2,
    WALL_HP = 3
} ResourceType;

typedef enum {
    NORMAL = 0,
    REPLAY = 1
} EnvMode;

// ========== STATIC DATA ARRAYS ==========
int DELTAS[4][2] = {
    {-1, 0}, // up
    {0, 1}, // right
    {1, 0}, // down
    {0, -1}, // left
};

// Offsets (row, col) for every direction
int ATTACK_SWORD[4][3][2] = {
    {{-1, -1}, {-1, 0}, {-1, 1}}, // up
    {{-1, 1}, {0, 1}, {1, 1}}, // right
    {{1, -1}, {1, 0}, {1, 1}}, // down
    {{-1, -1}, {0, -1}, {1, -1}}, // left
};

// ========== CORE DATA STRUCTURES ==========

// Required struct. Only use floats!
typedef struct {
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
    float births;
    float starvations;
    float murders;
    float stone_mined;
    float walls_built;
    float wall_destroyed;
    float food_stored;
    float food_eaten;
    float avg_population;
    float max_pop;
    float min_pop;
    float total_reward;
    float life_expectancy;
    float genetic_diversity;
} Log;

typedef struct {
    float births;
    float starvations;
    float murders;
    float stone_mined;
    float walls_built;
    float wall_destroyed;
    float food_stored;
    float food_eaten;
    float avg_population;
    float max_pop;
    float min_pop;
    float total_reward;
    float agent_life_expectancy;
    float agent_n;
} Stats;

typedef struct {
    int r; // row
    int c; // column
    Direction dir;
    int hp;
    int hp_max;
    int satiation;
    int max_satiation;
    int age;
    // inventory
    int food_carried;
    int stone_carried;
    int role;
} Agent;

// Client structure for rendering
typedef struct {
    Texture2D terrain_sprite;
    int* terrain_sprite_indices;
    Texture2D wall_sprite;
    int* wall_sprite_indices;
    Texture2D food_sprite;
    RenderTexture2D background_summer;  // Pre-rendered background
    RenderTexture2D background_winter;
    Camera2D camera;                    // Camera for world view
    Texture2D *char_bases;
    int max_crop_available;
    bool tracking_mode;
    int tracking_pid;
    bool is_paused;
} Client;

// Forward declarations
typedef struct Territories Territories;

// Agent Manager
typedef struct {
    Territories* env;            // Pointer to environment containing agents, kinship_matrix, and n_genes
    BitSet* alive_bitset;
    bool* alive_mask;                 // Track which slots are active [max_agents]
    unsigned short* free_pids;   // Stack of available PIDs
    unsigned short* alive_pids;  // List of consecutive alive PIDs
    unsigned short free_count;   // Number of free PIDs available
    unsigned short alive_count;  // Number of currently alive players
    unsigned short next_pid;     // For initial allocation
} AgentManager;

struct Territories {
    EnvMode render_mode;
    Client* client;
    Log log; // Required field. Env binding code uses this to aggregate logs
    Stats stats;
    float extinction_reward;
    Agent* agents;
    AgentManager* agent_manager;
    short* pids_2d; // -1 for empty, 0 to max_agents for agents
    bool* is_soil; // soil or grass
    unsigned short* tile_props; // amount of crop, stored food, stones and wall_hp
    unsigned char* observations; // using bytes to save memory
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // For when agents die
    unsigned char* truncations; // For when agents are alive but the episode is truncated
    unsigned char* alive_mask;
    unsigned char* kinship_matrix;
    unsigned char* dnas;
    unsigned short* family_sizes;
    unsigned short* prev_family_sizes;
    int n_genes;
    int n_alleles;
    int width;
    int height;
    int tick;
    bool is_winter;
    int obs_size;
    int max_agents;
    int n_roles;
    int min_ep_length;
    int max_ep_length;
    int next_max_ep_length;
    int* reserved_roles;
    char map_name[64];
    bool reward_growth_rate;
};

// ========== UTILITY FUNCTIONS ==========

unsigned char kinship_get(Territories* env, int pid1, int pid2, int n_genes) {
    unsigned char result = 0;
    for (int i = 0; i < n_genes; i++) {
        result += env->dnas[pid1*n_genes + i] == env->dnas[pid2*n_genes + i] ? 1 : 0;
    }
    return result;
}

void kinship_matrix_reset(Territories* env) {
    memset(env->kinship_matrix, 0, env->max_agents * env->max_agents * sizeof(unsigned char));
    for (int i = 0; i < env->max_agents; i++) {
        env->kinship_matrix[i*env->max_agents + i] = env->n_genes;
    }
}

void kinship_matrix_update(AgentManager* am, int pid) {
    unsigned char* kinship_matrix = am->env->kinship_matrix;
    am->env->prev_family_sizes[pid] = am->env->n_genes; // itself
    for (int pid2 = 0; pid2 < am->env->max_agents; pid2++) {
        if (!am->alive_mask[pid2] || pid2 == pid) continue; // Note we can't use alive_pids here because it's not updated yet
        unsigned char kinship = kinship_get(am->env, pid, pid2, am->env->n_genes);
        kinship_matrix[pid*am->env->max_agents + pid2] = kinship;
        kinship_matrix[pid2*am->env->max_agents + pid] = kinship;
        am->env->prev_family_sizes[pid] += kinship;
    }
    
} 

// ========== AGENT MANAGEMENT ==========

AgentManager* agent_manager_init(Territories* env) {
    AgentManager* am = (AgentManager*)calloc(1, sizeof(AgentManager));
    am->free_pids = (unsigned short*)calloc(env->max_agents, sizeof(unsigned short));
    am->alive_pids = (unsigned short*)calloc(env->max_agents, sizeof(unsigned short));
    am->alive_mask = (bool*)env->alive_mask;
    am->alive_bitset = bitset_create(env->max_agents);
    am->env = env;
    return am;
}

void agent_manager_reset(AgentManager* am) {
    memset(am->alive_mask, 0, am->env->max_agents * sizeof(bool));
    memset(am->free_pids, 0, am->env->max_agents * sizeof(unsigned short));
    memset(am->alive_pids, 0, am->env->max_agents * sizeof(unsigned short));
    bitset_clear(am->alive_bitset);
    am->free_count = 0;
    am->alive_count = 0;
    am->next_pid = 0;
}

int spawn_agent(AgentManager* am, int r, int c, int n_genes) {
    if (am->alive_count >= am->env->max_agents) return -1; // Cap reached
    int pid;
    if (am->free_count > 0) {
        // Reuse freed PID
        pid = am->free_pids[--am->free_count];
    } else {
        // Allocate new PID
        pid = am->next_pid++;
    }
    
    am->alive_mask[pid] = true;
    am->alive_count++;
    bitset_add(am->alive_bitset, pid);
    // Note: alive_pids array will be updated later via update_alive_pids()
    // Initialize agent
    Agent* agent = &am->env->agents[pid];
    agent->r = r;
    agent->c = c;
    agent->dir = (Direction)(rand() % 4); // Random direction between DIR_UP (0) and DIR_LEFT (3)
    agent->hp = 1;
    agent->hp_max = 1;
    agent->satiation = MAX_SATIATION;
    agent->max_satiation = MAX_SATIATION;
    agent->age = 0;
    agent->food_carried = 0;
    agent->stone_carried = 0;
    //agent->role = rand() % am->env->n_roles;
    
    return pid;
}

void kill_agent(AgentManager* am, int pid) {
    if (!am->alive_mask[pid]) return;
    
    am->alive_mask[pid] = false;
    am->free_pids[am->free_count++] = pid;  // Add to free pool
    am->alive_count--;
    bitset_remove(am->alive_bitset, pid);
    // Note: alive_pids array will be updated later via update_alive_pids()

    Agent *agent = &am->env->agents[pid];
    // if (agent->role == 0) {
    //     int dna_adr = 0;
    //     int multiplier = 1;
    //     for (int j = 0; j < am->env->n_genes; j++){
    //         dna_adr += multiplier * am->env->dnas[pid*am->env->n_genes + j];
    //         multiplier *= N_ALLELES;
    //     }
    //     am->env->reserved_roles[dna_adr] -= 1;
    // }
    am->env->stats.agent_life_expectancy += agent->age;
    am->env->stats.agent_n += 1;
}

void update_alive_pids(AgentManager* am) {
    int actual_count = bitset_update_members(am->alive_bitset, am->alive_pids);
    assert(actual_count == am->alive_count);
}

// ========== WORLD MECHANICS ==========

int get_growth_days(Territories* env, int r, int c) {
    if (env->is_winter || env->is_soil[r*env->width + c] == 0 ||
         env->tile_props[(r*env->width + c)*4 + STORED_FOOD] > 0 ||
         env->tile_props[(r*env->width + c)*4 + STONE] > 0 ||
         env->tile_props[(r*env->width + c)*4 + WALL_HP] > 0) return 0;
    int day_number = (env->tick + STARTING_DAY) % (SUMMER_DURATION + WINTER_DURATION);
    return min(day_number - env->tile_props[(r*env->width + c)*4 + LAST_HARVEST], MAX_GROWTH_DURATION);
}

void start_crop_growth(Territories* env) {
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            if (env->is_soil[r*env->width + c]) {
                env->tile_props[(r*env->width + c)*4 + LAST_HARVEST] = 0;
            }
        }
    }
} 

void update_wall_sprite_index(Territories* env, int r, int c, int R, int C) {
    if (env->tile_props[(r*C + c)*4 + WALL_HP] == 0) return;
    unsigned short (*tile_props_4d)[C][4] = (unsigned short(*)[C][4])env->tile_props;
    int up = (r-1+R)%R;
    int down = (r+1)%R;
    int left = (c-1+C)%C;
    int right = (c+1)%C;
    
    int tile_index = 0;
    if (tile_props_4d[up][c][WALL_HP] > 0) tile_index += 1;
    if (tile_props_4d[r][right][WALL_HP] > 0) tile_index += 2;
    if (tile_props_4d[down][c][WALL_HP] > 0) tile_index += 4;
    if (tile_props_4d[r][left][WALL_HP] > 0) tile_index += 8;
    env->client->wall_sprite_indices[r*C + c] = tile_index;
}

// Wall
void place_wall(Territories* env, int r, int c) {
    int R = env->height;
    int C = env->width;
    if (env->tile_props[(r*C + c)*4 + WALL_HP] > 0) return;
    // Clear any resources on this tile before placing wall
    for (int i = 0; i < 3; i++) { // First 3 properties are resources (crop, stored food, stone)
        env->tile_props[(r*C + c)*4 + i] = 0;
    }
    env->tile_props[(r*C + c)*4 + WALL_HP] = WALL_HP_MAX;
    if (env->client != NULL) {
        int up = (r-1+R)%R;
        int down = (r+1)%R;
        int left = (c-1+C)%C;
        int right = (c+1)%C;
        // update the tile sprite index and its neighbors
        update_wall_sprite_index(env, r, c, R, C);
        update_wall_sprite_index(env, up, c, R, C);
        update_wall_sprite_index(env, r, right, R, C);
        update_wall_sprite_index(env, down, c, R, C);
        update_wall_sprite_index(env, r, left, R, C);
    }
}

void destroy_wall(Territories* env, int r, int c) {
    if (env->tile_props[(r*env->width + c)*4 + WALL_HP] == 0) return;
    env->tile_props[(r*env->width + c)*4 + WALL_HP] = 0;
    int day_number = (env->tick + STARTING_DAY) % (SUMMER_DURATION + WINTER_DURATION);
    if (!env->is_winter && env->is_soil[r*env->width + c]) {
        // the crop can start to grow again, once the wall is destroyed
        env->tile_props[(r*env->width + c)*4 + LAST_HARVEST] = day_number;
    }
    if (env->client != NULL) {
        int R = env->height;
        int C = env->width;
        int up = (r-1+R)%R;
        int down = (r+1)%R;
        int left = (c-1+C)%C;
        int right = (c+1)%C;
        // update the tile sprite index and its neighbors
        update_wall_sprite_index(env, r, c, R, C);
        update_wall_sprite_index(env, up, c, R, C);
        update_wall_sprite_index(env, r, right, R, C);
        update_wall_sprite_index(env, down, c, R, C);
        update_wall_sprite_index(env, r, left, R, C);
    }
}

bool tile_is_blocked(Territories* env, int r, int c) {
    return env->tile_props[(r*env->width + c)*4 + WALL_HP] > 0 || 
    env->tile_props[(r*env->width + c)*4 + STONE] > 0 ||
    env->pids_2d[r*env->width + c] != -1;
}

// ========== AGENT ACTIONS ==========


bool _agent_can_reproduce(Agent *agent) {
    return agent->age >= REPRODUCTION_AGE && agent->satiation > MAX_SATIATION/2;
}

int find_empty_cell(int r, int c, Territories* env) {
    int R = env->height;
    int C = env->width;
    for (int r_offset = -1; r_offset <= 1; r_offset++) {
        for (int c_offset = -1; c_offset <= 1; c_offset++) {
            if (r_offset == 0 && c_offset == 0) continue; // skip self
            int tr = (r + r_offset + R) % R;
            int tc = (c + c_offset + C) % C;
            if (!tile_is_blocked(env, tr, tc)) {
                return tr*C + tc;
            }
        }
    }
    return -1;
}

int _find_mate(Agent* agent, Territories* env) {
    int R = env->height;
    int C = env->width;
    int r = agent->r;
    int c = agent->c;
    for (int r_offset = -1; r_offset <= 1; r_offset++) {
        for (int c_offset = -1; c_offset <= 1; c_offset++) {
            if (r_offset == 0 && c_offset == 0) continue; // skip self
            int tr = (r + r_offset + R) % R;
            int tc = (c + c_offset + C) % C;
            int pid = env->pids_2d[tr*C + tc];
            if (pid != -1 && env->actions[pid] == REPRODUCE) {
                    Agent* mate = &env->agents[pid];
                    if (_agent_can_reproduce(mate)){
                        return pid;
                    }    
            }
        }
    }
    return -1;
}

void agent_reproduce(int pid, Territories* env) {
    Agent* agent = &env->agents[pid];
    if (!_agent_can_reproduce(agent) || 
    (env->agent_manager->alive_count >= env->max_agents)) return;
    int mate_pid = _find_mate(agent, env);
    if (mate_pid == -1) return;
    int empty_adr = find_empty_cell(agent->r, agent->c, env);
    Agent* mate = &env->agents[mate_pid];
    agent->satiation -= MAX_SATIATION/2;
    mate->satiation -= MAX_SATIATION/2;

    int new_r = empty_adr / env->width;
    int new_c = empty_adr % env->width;
    int child_pid = spawn_agent(env->agent_manager, new_r, new_c, env->n_genes);
    env->pids_2d[new_r*env->width + new_c] = child_pid;

    for (int j = 0; j < env->n_genes; j++){
        int p = rand() % 2;
        env->dnas[child_pid*env->n_genes + j] = p == 0 ? env->dnas[pid*env->n_genes + j] : env->dnas[mate_pid*env->n_genes + j];
    }
    Agent* child = &env->agents[child_pid];
    child->role = rand() % env->n_roles;
    // if (env->reserved_roles[dna_adr] < 2) {
    //     child->role = 0;
    //     env->reserved_roles[dna_adr] += 1;
    // }
    // else {
    //     child->role = rand() % (env->n_roles - 1) + 1;
    // }
    // this runs always after the dna is assigned
    kinship_matrix_update(env->agent_manager, child_pid);
    if (env->tick < env->min_ep_length) {
        env->stats.births += 1;
    }
}




void agent_mine(Agent *agent, Territories* env) {
    if (agent->stone_carried >= MAX_STONE_CARRYING_CAPACITY) return;
    int R = env->height;
    int C = env->width;
    unsigned short (*tile_props_3d)[env->width][4] = (unsigned short(*)[env->width][4])env->tile_props;
    int target_r = -1;
    int target_c = -1;
    for (int dir = 0; dir < 4; dir++) {
        int rr = (agent->r + DELTAS[dir][0] + R) % R;
        int cc = (agent->c + DELTAS[dir][1] + C) % C;
        if (tile_props_3d[rr][cc][STONE] > 0) {
            target_c = cc;
            target_r = rr;
            agent->dir = dir;
            break;
        }
    }
    if (target_c != -1) {
        tile_props_3d[target_r][target_c][STONE] -= 1;
        agent->stone_carried += 1;
        if (env->tick < env->min_ep_length) {
            env->stats.stone_mined += STONE_MINED_PER_TICK;
        }
    }
}

void agent_attack(Agent *agent, Territories* env) {
    int r = agent->r;
    int c = agent->c;
    int R = env->height;
    int C = env->width;
    int (*deltas)[3][2] = (int(*)[3][2])ATTACK_SWORD;
    unsigned short (*tile_props_3d)[env->width][4] = (unsigned short(*)[env->width][4])env->tile_props;

    int target_r;
    int target_c;
    bool is_wall = false;
    bool hit = false;
    for (int direction = 0; direction < 4; direction++) {
        int dir = (agent->dir + direction) % 4;
        
        int dr = deltas[dir][0][0];
        int dc = deltas[dir][0][1];
        int rr = (r + dr + R) % R;
        int cc = (c + dc + C) % C;
        
        if (tile_props_3d[rr][cc][WALL_HP] > 0) {
            is_wall = true;
            hit = true;
        }
        else if (env->pids_2d[rr*env->width + cc] != -1){
            hit = true;
        }
        if (hit) {
            agent->dir = dir;
            target_r = rr;
            target_c = cc;
            break;
        }
    }
    if (!hit) return;
    if (is_wall) {
        tile_props_3d[target_r][target_c][WALL_HP] -= 1;
        if (tile_props_3d[target_r][target_c][WALL_HP] == 0) {
            if (env->tick < env->min_ep_length) {
                env->stats.wall_destroyed += 1;
            }
            destroy_wall(env, target_r, target_c);
        }
    } else {
        int target_pid = env->pids_2d[target_r*env->width + target_c];
        Agent* target = &env->agents[target_pid];
        target->hp -= 1;
        if (target->hp == 0) {
            // loot resources
            if (env->tick < env->min_ep_length) {
                env->stats.murders += 1;
            }
            agent->satiation = min(MAX_SATIATION, agent->satiation + target->satiation/2);
            agent->stone_carried = min(MAX_STONE_CARRYING_CAPACITY, agent->stone_carried + target->stone_carried);
            agent->food_carried = min(MAX_FOOD_CARRYING_CAPACITY, agent->food_carried + target->food_carried);
            // kill_agent(env->agent_manager, target_pid);
            // env->pids_2d[target_r*env->width + target_c] = -1;
            // env->terminals[target_pid] = 1;            
        }
    }
}

// ========== OBSERVATION AND REWARDS ==========

void _delta_rewards(Territories* env) {
    AgentManager* am = env->agent_manager;
    unsigned char (*kinship_matrix_2d)[env->max_agents] = (unsigned char(*)[env->max_agents])env->kinship_matrix;
    for (int pid = 0; pid < env->max_agents; pid++) {
        if (am->alive_mask[pid] || env->terminals[pid]) {
            env->family_sizes[pid] = 0;
            for (int j = 0; j < am->alive_count; j++) {
                int pid2 = am->alive_pids[j];
                env->family_sizes[pid] += kinship_matrix_2d[pid][pid2];
            }
            env->rewards[pid] = ((float)env->family_sizes[pid] - (float)env->prev_family_sizes[pid]) / ((float) env->n_genes);
            if (env->tick < env->min_ep_length) {
                env->stats.total_reward += env->rewards[pid];
            }
       }
   }
   memcpy(env->prev_family_sizes, env->family_sizes, env->max_agents * sizeof(unsigned short));
}

void _growth_rate_rewards(Territories* env) {
    AgentManager* am = env->agent_manager;
    unsigned char (*kinship_matrix_2d)[env->max_agents] = (unsigned char(*)[env->max_agents])env->kinship_matrix;
    for (int pid = 0; pid < env->max_agents; pid++) {
        if (am->alive_mask[pid] || env->terminals[pid]) {
            env->family_sizes[pid] = 0;
            for (int j = 0; j < am->alive_count; j++) {
                int pid2 = am->alive_pids[j];
                env->family_sizes[pid] += kinship_matrix_2d[pid][pid2];
            }
            if (env->family_sizes[pid] == 0) {
                assert(env->terminals[pid]);
                env->rewards[pid] = env->extinction_reward;
                if (env->prev_family_sizes[pid] > 1) {
                    // If a family goes from N -> 0, with N > 1, the reward should be log(1/N) + extinction reward
                    // Otherwise, it is better to go extinct all at once, then one at a time.
                    env->rewards[pid] += log(1.0f/(float)env->prev_family_sizes[pid]);
                }
            } else {
                env->rewards[pid] = log((float)env->family_sizes[pid]/(float)env->prev_family_sizes[pid]);
            }
            if (env->tick < env->min_ep_length) {
                env->stats.total_reward += env->rewards[pid];
            }
       }
   }
   memcpy(env->prev_family_sizes, env->family_sizes, env->max_agents * sizeof(unsigned short));
}

void compute_rewards(Territories* env) {
     if (env->reward_growth_rate) {
        _growth_rate_rewards(env);
     } else {
        _delta_rewards(env);
     }
}

void compute_all_obs(Territories* env) {
    AgentManager* am = env->agent_manager;
    unsigned char* obs = env->observations;
    for (int i = 0; i < am->alive_count; i++) {
        int pid = am->alive_pids[i];
        int obs_adr = pid * env->obs_size;

        unsigned short (*tile_props_3d)[env->width][4] = (unsigned short(*)[env->width][4])env->tile_props;
        Agent* agent = &env->agents[pid];
        int r = agent->r;
        int c = agent->c;
        // Vision
        for (int r_offset = -VISION_RADIUS; r_offset <= VISION_RADIUS; r_offset++) {
            for (int c_offset = -VISION_RADIUS; c_offset <= VISION_RADIUS; c_offset++) {
                int tr = (r + r_offset + env->height) % env->height;
                int tc = (c + c_offset + env->width) % env->width;
                // Terrain
                obs[obs_adr] = env->is_soil[tr*env->width + tc] ? 1.0 : 0.0;
                obs[obs_adr+1] = get_growth_days(env, tr, tc);
                obs[obs_adr+2] = float_to_byte((float)tile_props_3d[tr][tc][STORED_FOOD], 0, MAX_FOOD_STORAGE_CAPACITY);
                obs[obs_adr+3] = float_to_byte((float)tile_props_3d[tr][tc][STONE], 0, STONE_PER_MINE);
                obs[obs_adr+4] = float_to_byte((float)tile_props_3d[tr][tc][WALL_HP], 0, WALL_HP_MAX);
                
                // Agents
                int pid2 = env->pids_2d[tr*env->width + tc];
                if (pid2 != -1) {
                    Agent* agent2 = &env->agents[pid2];
                    obs[obs_adr+5] = float_to_byte(kinship_get(env, pid, pid2, env->n_genes), 0, 1.0f);
                    obs[obs_adr+6] = float_to_byte((float)agent2->hp, 0, MAX_HP);
                    obs[obs_adr+7] = float_to_byte((float)agent2->age, 0, 100);
                    obs[obs_adr+8] = float_to_byte((float)agent2->satiation, 0, MAX_SATIATION);
                    obs[obs_adr+9] = agent2->dir + 1; // 0 is no agent
                    obs[obs_adr+10] = agent2->role + 1; // 0 is no agent
                    for (int i = 0; i < env->n_genes; i++) {
                        obs[obs_adr+11+i] = (float)env->dnas[pid2*env->n_genes + i] + 1; // 0 is no agent
                    }
                } else {
                    for (int i = 0; i < 6; i++) {
                        obs[obs_adr+5+i] = 0;
                    }
                    for (int i = 0; i < env->n_genes; i++) {
                        obs[obs_adr+11+i] = 0;
                    }
                }
                obs_adr += 11 + env->n_genes;
            }
        }
        // Self Information
        obs[obs_adr] = float_to_byte((float)agent->food_carried, 0, MAX_FOOD_CARRYING_CAPACITY);
        obs[obs_adr+1] = float_to_byte((float)agent->stone_carried, 0, MAX_STONE_CARRYING_CAPACITY);
        obs[obs_adr+2] = float_to_byte((float)agent->hp, 0, MAX_HP);
        obs[obs_adr+3] = float_to_byte((float)agent->satiation, 0, MAX_SATIATION);
        obs[obs_adr+4] = float_to_byte((float)agent->age, 0, 100);
        obs[obs_adr+5] = (float)agent->role;
        for (int i = 0; i < env->n_genes; i++) {
            obs[obs_adr+6+i] = (float)env->dnas[pid*env->n_genes + i];
        }

        obs_adr += 6 + env->n_genes;
        // Cultural Knowledge
        obs[obs_adr]   = float_to_byte((float)agent->r, 0, env->height);
        obs[obs_adr+1] = float_to_byte((float)agent->c, 0, env->width);
        int day_number = (env->tick + STARTING_DAY) % (SUMMER_DURATION + WINTER_DURATION);
        obs[obs_adr+2] = float_to_byte((float)(day_number), 0, SUMMER_DURATION + WINTER_DURATION);
        obs[obs_adr+3] = float_to_byte(env->family_sizes[pid], 0, env->max_agents);
        obs[obs_adr+4] = float_to_byte((float)env->agent_manager->alive_count, 0, env->max_agents);
        assert(obs_adr + 5 == env->obs_size * (pid+1));
    }
}

// ========== CORE ENVIRONMENT FUNCTIONS ==========

/* Recommended to have an init function of some kind if you allocate 
* extra memory. This should be freed by c_close. Don't forget to call
* this in binding.c!
*/
void init(Territories* env) {
    // Arrays (observations, actions, rewards, terminals) should be allocated before calling init:
    // - For standalone C: allocated in main()
    // - For Python binding: provided by binding
    assert(env->n_genes < 4 && env->n_genes >= 0); // we can't yet handle rendering more than 3 genes
    assert(env->max_agents > 0);
    assert(env->n_roles > 0);
    assert(env->min_ep_length > 0);
    assert(env->min_ep_length < env->max_ep_length);
    assert(env->extinction_reward < 0);
    assert(env->n_alleles > 0);
    
    env->obs_size = (2*VISION_RADIUS+1)*(2*VISION_RADIUS+1)*(11+env->n_genes)+6+env->n_genes+5;
    env->agents = (Agent*)calloc(env->max_agents, sizeof(Agent));
    // env->reserved_roles = (int*)calloc((pow(N_ALLELES, env->n_genes)-1), sizeof(int));
    env->family_sizes = (unsigned short*)calloc(env->max_agents, sizeof(unsigned short));
    env->prev_family_sizes = (unsigned short*)calloc(env->max_agents, sizeof(unsigned short));

    // Create map
    env->tile_props = (unsigned short*)calloc(env->width * env->height * 4, sizeof(unsigned short)); // 4 types of tile properties: crop, stored food, stone, wall_hp
    env->is_soil = read_is_soil(env->width, env->height, env->map_name);
    
    // Initialize pids arrays
    env->pids_2d = (short*)calloc(env->width * env->height, sizeof(short));
    env->agent_manager = agent_manager_init(env);
}

float compute_genetic_diversity(Territories* env) {
    AgentManager* am = env->agent_manager;
    float diversity = 0;
    int n_alleles = env->n_alleles;
    if (am->alive_count == 0) return 0.0f;
    int *allele_counts = calloc(env->n_genes * n_alleles , sizeof(int));

    for (int i = 0; i < am->alive_count; i++) {
        int pid = am->alive_pids[i];
        for (int j = 0; j < env->n_genes; j++) {
            allele_counts[j*n_alleles + env->dnas[pid*env->n_genes + j]] += 1;
        }
    }

    for (int l = 0; l < env->n_genes; l++) {
        for (int i = 0; i < n_alleles; i++) {
            if (allele_counts[l*n_alleles + i] == 0) continue;
            float prob = (float)allele_counts[l*n_alleles + i] / (float)am->alive_count;
            diversity += -prob * log2(prob);
        }
    }
    free(allele_counts);
    return diversity;
}

void update_episode_logs(Territories* env) {
    env->log.births += env->stats.births;
    env->log.starvations += env->stats.starvations;
    env->log.murders += env->stats.murders;
    env->log.stone_mined += env->stats.stone_mined;
    env->log.walls_built += env->stats.walls_built;
    env->log.wall_destroyed += env->stats.wall_destroyed;
    env->log.food_stored += env->stats.food_stored;
    env->log.food_eaten += env->stats.food_eaten;
    env->log.max_pop += env->stats.max_pop;
    env->log.min_pop += env->stats.min_pop;
    env->log.avg_population += env->stats.avg_population / (float)(min(env->tick, env->min_ep_length));
    env->log.total_reward += env->stats.total_reward;
    env->stats.births = 0;
    env->stats.starvations = 0;
    env->stats.murders = 0;
    env->stats.stone_mined = 0;
    env->stats.walls_built = 0;
    env->stats.wall_destroyed = 0;
    env->stats.food_stored = 0;
    env->stats.food_eaten = 0;
    env->stats.max_pop = 0;
    env->stats.min_pop = 0;
    env->stats.avg_population = 0;
    env->stats.total_reward = 0;
    env->log.n += 1;
    env->log.episode_length += env->tick;
    AgentManager* am = env->agent_manager;
    if (am->alive_count > 0) {
        for (int i = 0; i < am->alive_count; i++) {
            int pid = am->alive_pids[i];
            env->stats.agent_life_expectancy += env->agents[pid].age;
            env->stats.agent_n += 1;
        }
    }
    env->log.genetic_diversity += compute_genetic_diversity(env);
    env->log.life_expectancy =  env->stats.agent_n > 0 ? env->stats.agent_life_expectancy / env->stats.agent_n : 0;
    env->stats.agent_life_expectancy = 0;
    env->stats.agent_n = 0;
}

void c_reset(Territories* env) {
    memset(env->truncations, 0, env->max_agents * sizeof(unsigned char));
    // memset(env->reserved_roles, 0, (pow(N_ALLELES, env->n_genes)-1) * sizeof(int));
    memset(env->prev_family_sizes, 0, env->max_agents * sizeof(unsigned short));
    env->tick = 0;
    env->is_winter = false;
    env->next_max_ep_length = env->min_ep_length + rand() % (env->max_ep_length - env->min_ep_length);

    AgentManager* am = env->agent_manager;
    memset(env->pids_2d, -1, env->width * env->height * sizeof(short));
    assert(env->pids_2d[0] == -1); // in some platforms, memset does not work with negative values.
    // Clean up agents
    if (am->alive_count > 0) {
        int initial_alive_count = am->alive_count; // Alive count will be modified during the loop
        for (int i = 0; i < initial_alive_count; i++) {
            int pid = am->alive_pids[i];
            kill_agent(am, pid);
        }
        update_alive_pids(env->agent_manager); 
        assert(am->alive_count == 0);
    }
    // Clean up terrain
    memset(env->tile_props, 0, env->width * env->height * 4 * sizeof(unsigned short));

    // Spawn stones
    int C = env->width;
    int R = env->height;
    for (int ri = 0; ri < 2; ri++) {
        for (int ci = 0; ci < 2; ci++) {
            int r = (int)((0.25 + ri * 0.50) * R);
            int c = (int)((0.25 + ci * 0.50) * C);
            env->tile_props[(r*C + c)*4 + STONE] = STONE_PER_MINE;
        }
    }
    env->tile_props[((int)(R/2)*C + (int)(C/2))*4 + STONE] = STONE_PER_MINE;

    // Place initial agents
    for (int i = 0; i < env->width * env->height; i++) {
        env->pids_2d[i] = -1;
    }
    kinship_matrix_reset(env);
    agent_manager_reset(env->agent_manager);
    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 3; j++) {
    //         while (true) {
    //             int adr = rand() % (env->width * env->height);
    //             int r = adr / env->width;
    //             int c = adr % env->width;
    //             if (tile_is_blocked(env, r, c)) continue;
    //             int pid = spawn_agent(env->agent_manager, r, c, env->n_genes);
    //             env->pids_2d[r*env->width + c] = pid;
    //             env->agents[pid].role = rand() % env->n_roles;
    //             for (int g = 0; g < env->n_genes; g++) {
    //                 env->dnas[pid*env->n_genes + g] = rand() % env->n_alleles;
    //             }
    //             // env->dnas[pid*env->n_genes + 0] = i & 1;
    //             // env->dnas[pid*env->n_genes + 1] = (i >> 1) & 1;
    //             env->agents[pid].age = REPRODUCTION_AGE;
    //             env->agents[pid].hp_max = MAX_HP;
    //             env->agents[pid].hp = MAX_HP;
    //             kinship_matrix_update(env->agent_manager, pid);
    //             break;
    //         }
    //     }
    // }
    for (int i = 0; i < 4; i++) {
        while (true) {
            int adr = rand() % (env->width * env->height);
            int r = adr / env->width;
            int c = adr % env->width;
            if (tile_is_blocked(env, r, c)) continue;
            int n_adr = find_empty_cell(r, c, env);
            if (n_adr == -1) continue;
            int r_2 = n_adr / env->width;
            int c_2 = n_adr % env->width;
            int pid = spawn_agent(env->agent_manager, r, c , env->n_genes);
            env->pids_2d[r*env->width + c] = pid;
            int pid2 = spawn_agent(env->agent_manager, r_2, c_2, env->n_genes);
            env->pids_2d[r_2*env->width + c_2] = pid2;
            for (int j = 0; j < env->n_genes; j++){
                int allele = rand() % env->n_alleles;
                env->dnas[pid*env->n_genes + j] = allele;
                env->dnas[pid2*env->n_genes + j] = allele;                
            }
            Agent* agent1 = &env->agents[pid];
            Agent* agent2 = &env->agents[pid2];
            agent1->role = 0;
            agent2->role = 0;
            kinship_matrix_update(env->agent_manager, pid);
            kinship_matrix_update(env->agent_manager, pid2);
            break;
        }
    }
    // Ensure alive_pids is current after all the births
    update_alive_pids(env->agent_manager); 

    // We're computing rewards here because we need to know the family sizes to compute the observations.
    compute_rewards(env);
    compute_all_obs(env);
}

// Required function
void c_step(Territories* env) {
    // Timeline:
    //  0. Tick
    //  1. Agents eat and act
    //  2. Agents (including children) observe 

    // Kill agents that starved or were killed in the previous step
    // note we just kill them now because we had to output their terminals, dna and alive.

    memset(env->terminals, 0, env->max_agents * sizeof(unsigned char));
    AgentManager* am = env->agent_manager;
    if (am->alive_count == 0 || env->tick >= env->next_max_ep_length) {
        // either all dead or episode max length reached - reset the environment
        update_episode_logs(env);
        c_reset(env);
        return;
    }

    int R = env->height;
    int C = env->width;
    int day_number = (env->tick + STARTING_DAY) % (SUMMER_DURATION + WINTER_DURATION);
    unsigned short (*tile_props_3d)[env->width][4] = (unsigned short(*)[env->width][4])env->tile_props;

    if (!env->is_winter && day_number >= SUMMER_DURATION) { // summer from 0 to 99
        env->is_winter = true;
    }
    else if (env->is_winter && day_number < SUMMER_DURATION) { // winter from 100 to 199
        env->is_winter = false;
        start_crop_growth(env);
    }
    env->tick++;
    if (env->tick < env->min_ep_length) {
        env->stats.avg_population += am->alive_count;
        env->stats.max_pop = max(env->stats.max_pop, am->alive_count);
        env->stats.min_pop = min(env->stats.min_pop, am->alive_count);
    }
    // randomise the order in which actions are taken
    shuffle(am->alive_pids, am->alive_count);

    // Agents act
    int initial_alive_count = am->alive_count; // Alive count will be modified during the loop
    for (int i = 0; i < initial_alive_count; i++) {
        unsigned short pid = am->alive_pids[i];
        Agent* agent = &env->agents[pid];
        if (agent->hp <= 0) continue; // agent was killed this step
        assert(agent->satiation > 0);
        agent->age++;
        if (agent->age == REPRODUCTION_AGE) {
            agent->hp_max = MAX_HP;
            agent->hp = MAX_HP;
        }
        agent->satiation -= METABOLISM_RATE;
        // Agents eat
        if (agent->food_carried > 0) {
            int appetite = MAX_SATIATION - agent->satiation;
            int food_to_eat = min(appetite, agent->food_carried);
            agent->food_carried -= food_to_eat;
            agent->satiation += food_to_eat;
            if (env->tick < env->min_ep_length) {
                env->stats.food_eaten += food_to_eat;
            }
        }

        int action = env->actions[pid];
        if (action >= MOVE_UP && action <= MOVE_LEFT) { // move
            int new_r = agent->r;
            int new_c = agent->c;
            if (action == (int)agent->dir) { // actually move
                if (action == MOVE_UP) {
                    new_r--;
                }
                else if (action == MOVE_RIGHT) {
                    new_c++;
                }
                else if (action == MOVE_DOWN) {
                    new_r++;
                }
                else {
                    new_c--;
                }
                new_r = (new_r + R) % R;
                new_c = (new_c + C) % C;
                if (!tile_is_blocked(env, new_r, new_c)) {
                    env->pids_2d[agent->r*C + agent->c] = -1;
                    env->pids_2d[new_r*C + new_c] = pid;
                    agent->r = new_r;
                    agent->c = new_c;
                }
            }
            agent->dir = action;
        }
        else if (action == PICKUP) {
            int food_capacity = MAX_FOOD_CARRYING_CAPACITY - agent->food_carried;
            if (tile_props_3d[agent->r][agent->c][STORED_FOOD] > 0) {
                int stored_food = tile_props_3d[agent->r][agent->c][STORED_FOOD];
                int food_to_pickup = min(stored_food, food_capacity);
                tile_props_3d[agent->r][agent->c][STORED_FOOD] -= food_to_pickup;
                agent->food_carried += food_to_pickup;
                if (food_to_pickup == stored_food && !env->is_winter && env->is_soil[agent->r * C + agent->c]) {
                    // the tile can start to grow crop again, once it is no longer storing food
                    tile_props_3d[agent->r][agent->c][LAST_HARVEST] = day_number;
                }
            }
            else {
                int growth_days = get_growth_days(env, agent->r, agent->c);
                if (growth_days > 0) {
                    int crop_available = (int)(exp(K * growth_days) - 1);
                    int food_to_pickup = min(crop_available, food_capacity);
                    tile_props_3d[agent->r][agent->c][LAST_HARVEST] = day_number;
                    agent->food_carried += food_to_pickup;
                    if (food_to_pickup < crop_available) {
                        // remaining food is stored on the tile
                        tile_props_3d[agent->r][agent->c][STORED_FOOD] = crop_available - food_to_pickup;
                    }
                }
            }
        }
        else if (action == MINE) {
            agent_mine(agent, env);
        }
        else if (action == PACKAGE_FOOD) {
            // package food being carried or/and crop on the tile
            int growth_days = get_growth_days(env, agent->r, agent->c);
            if (growth_days > 0) { // if there is crop, store it
                int crop_available = (int)(exp(K * growth_days) - 1);
                tile_props_3d[agent->r][agent->c][LAST_HARVEST] = day_number;
                tile_props_3d[agent->r][agent->c][STORED_FOOD] += crop_available;
                if (env->tick < env->min_ep_length) {
                    env->stats.food_stored += crop_available;
                }
            }
            if (agent->food_carried > 0) {
                int tile_storage_capacity = MAX_FOOD_STORAGE_CAPACITY - tile_props_3d[agent->r][agent->c][STORED_FOOD];
                int food_to_drop = min(agent->food_carried, tile_storage_capacity);
                tile_props_3d[agent->r][agent->c][STORED_FOOD] += food_to_drop;
                agent->food_carried -= food_to_drop;
                if (env->tick < env->min_ep_length) {
                    env->stats.food_stored += food_to_drop;
                }
            }
        }
        else if (action == BUILD_WALL) {
            if (agent->stone_carried > 0) {
                // build wall in the direction of the agent
                int wall_r = (agent->r + DELTAS[agent->dir][0] + R) % R;
                int wall_c = (agent->c + DELTAS[agent->dir][1] + C) % C;
                if (!tile_is_blocked(env, wall_r, wall_c)) {
                    place_wall(env, wall_r, wall_c);
                    agent->stone_carried -= 1;
                    if (env->tick < env->min_ep_length) {
                        env->stats.walls_built += 1;
                    }
                }
            }
        }
        else if (action == ATTACK) {
            agent_attack(agent, env);
        }
        else if (action == REPRODUCE) {
            agent_reproduce(pid, env);
        }
        
        if (agent->satiation <= 0) {
            if (env->tick < env->min_ep_length) {
                env->stats.starvations += 1;
            }
        }
    }
    // Ensure alive_pids is current after all the births
    update_alive_pids(am); 

    // Kill all agents at the end of the step.
    // Note:
    // - their alive mask is updated
    // - their terminals are set
    // - the kinship matrix is not updated - it only changes when agents are born.
    // This allows us to compute the final reward and family value for those who died.
    initial_alive_count = am->alive_count; // Alive count will be modified during the loop
    for (int i = 0; i < initial_alive_count; i++) {
        unsigned short pid = am->alive_pids[i];
        Agent* agent = &env->agents[pid];
        if (agent->satiation <= 0 || agent->hp <= 0) {
            kill_agent(am, pid);
            env->pids_2d[agent->r*C + agent->c] = -1;
            env->terminals[pid] = 1;
        }
    }
    // Ensure alive_pids is up-to-date after all the deaths
    update_alive_pids(am); 

    if (env->tick >= env->next_max_ep_length) {
        memset(env->truncations, 1, env->max_agents * sizeof(unsigned char));
    }
    compute_rewards(env);
    compute_all_obs(env);
}

// ========== RENDERING SYSTEM ==========

void fill_tiles_sprite_indices(bool* flat_is_soil, int* flat_tiles_sprite_indices, int R, int C) {
    // https://excaliburjs.com/blog/Autotiling%20Technique/
    // wang 2 index system: https://dev.to/joestrout/wang-2-corner-tiles-544k
    // sprite sheet has two rows of tiles: one for summer, one for winter
    // 0 - 4: variations of grass, 5 - 14: wang transition tiles, 15 - 19: variations of soil
    bool (*is_soil)[C] = (bool(*)[C])flat_is_soil;
    int (*tiles_sprite_indices)[C] = (int(*)[C])flat_tiles_sprite_indices;
    for (int r = 0; r < R; r++) {
        int up = (r-1+R)%R;
        int down = (r+1)%R;
        
        for (int c = 0; c < C; c++) {
            int left = (c-1+C)%C;
            int right = (c+1)%C;
            int adr = 0;
            
            bool current_is_soil = is_soil[r][c];

            // check corners
            if (current_is_soil) {
                bool top_right = is_soil[up][c] && is_soil[up][right] && is_soil[r][right];
                bool bottom_right = is_soil[r][right] && is_soil[down][right] && is_soil[down][c];
                bool bottom_left = is_soil[r][left] && is_soil[down][left] && is_soil[down][c];
                bool top_left = is_soil[r][left] && is_soil[up][left] && is_soil[up][c];
              
                if (top_right) adr += 1;
                if (bottom_right) adr += 2;
                if (bottom_left) adr += 4;
                if (top_left) adr += 8;
                if (adr == 0) adr += 15 + rand() % 5;
                
                adr += 4;
            } else {
                adr += rand() % 5;
            }

            tiles_sprite_indices[r][c] = adr;
        }
    }
}


void fill_the_background(Territories* env, Client* client, RenderTexture2D target, bool isWinter) {
    BeginTextureMode(target);
    ClearBackground(BLANK);
    
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            int adr = r*env->width + c;
            int tile = client->terrain_sprite_indices[adr];
            int u = TILE_SIZE*(tile % 24);
            int v = isWinter ? TILE_SIZE : 0;
            
            Vector2 dest_pos = {
                .x = c*TILE_SIZE,
                .y = r*TILE_SIZE, 
            };

            Rectangle source_rect = (Rectangle){
                .x = u,
                .y = v,
                .width = TILE_SIZE,
                .height = TILE_SIZE,
            };

            DrawTextureRec(client->terrain_sprite, source_rect, dest_pos, WHITE);
        }
    }
    
    EndTextureMode();
}

Client* make_client(Territories* env) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Territories");
    SetTargetFPS(FRAME_RATE);
    Client* client = (Client*)calloc(1, sizeof(Client));

    // Load textures
    client->terrain_sprite = LoadTexture("resources/terrain_64_64.png");
    client->food_sprite = LoadTexture("resources/food_64_64.png");
    client->wall_sprite = LoadTexture("resources/stone_wall_64_64.png");
    client->wall_sprite_indices = (int*)calloc(env->width * env->height, sizeof(int));
    client->terrain_sprite_indices = (int*)calloc(env->width * env->height, sizeof(int));
    fill_tiles_sprite_indices(env->is_soil, client->terrain_sprite_indices, env->height, env->width);
    client->background_summer = LoadRenderTexture(TILE_SIZE * env->width, TILE_SIZE * env->height);
    client->background_winter = LoadRenderTexture(TILE_SIZE * env->width, TILE_SIZE * env->height);
    fill_the_background(env, client, client->background_summer, false);
    fill_the_background(env, client, client->background_winter, true);

    client->char_bases = (Texture2D*)calloc(env->n_alleles * env->n_genes, sizeof(Texture2D));
    char char_filename[64];
    for (int i = 0; i < env->n_genes; i++) {
        for (int a = 0; a < env->n_alleles ; a++) {
            if (i > 0 && a == 3){
                client->char_bases[i*env->n_alleles + a].id = 0; // empty texture
                continue;
            }
            if (i == 0) {
                snprintf(char_filename, sizeof(char_filename), "resources/char/char_a_p1_0bas_humn_v0%d_128.png", a);
            } else if (i == 1) {
                snprintf(char_filename, sizeof(char_filename), "resources/char/char_a_p1_4har_bob1_v0%d_128.png", a);
            } else {
                assert(i == 2);
                snprintf(char_filename, sizeof(char_filename), "resources/char/char_a_p1_1out_pfpn_v0%d_128.png", a);
            }           
            client->char_bases[i*env->n_alleles + a] = LoadTexture(char_filename);
        }
    }
    
    // Initialize camera
    client->camera.target = (Vector2){ TILE_SIZE * env->width / 2.0f, TILE_SIZE * env->height / 2.0f }; // where the camera is looking
    client->camera.offset = (Vector2){ GAME_WIDTH / 2.0f, GAME_HEIGHT / 2.0f }; // where the target is on the camera
    client->camera.rotation = 0.0f;
    client->camera.zoom = 1.0f;

    client->max_crop_available = (int)(exp(K * MAX_GROWTH_DURATION) - 1);
    
    return client;
}

void control_camera_zoom(Client* client) {
        float delta_time = GetFrameTime();
        // Zoom with + and - keys (both main keyboard and keypad)
        float zoom_speed = 2.0f; // zoom factor per second
        if (IsKeyDown(KEY_EQUAL) || IsKeyDown(KEY_KP_ADD)) { // + key or keypad +
            client->camera.zoom += zoom_speed * delta_time;
        }
        if (IsKeyDown(KEY_MINUS) || IsKeyDown(KEY_KP_SUBTRACT)) { // - key or keypad -
            client->camera.zoom -= zoom_speed * delta_time;
        }
        
        // Calculate world dimensions
        float world_width = client->background_summer.texture.width;
        float world_height = client->background_summer.texture.height;
        
        // Calculate minimum zoom to prevent showing black space on multiple sides
        float min_zoom_x = (float)GAME_WIDTH / world_width;   // Zoom needed to fit width
        float min_zoom_y = (float)GAME_HEIGHT / world_height; // Zoom needed to fit height
        float min_zoom = fminf(min_zoom_x, min_zoom_y);         // Use the less restrictive one
        
        // Clamp zoom to reasonable limits
        client->camera.zoom = fmaxf(min_zoom, fminf(5.0f, client->camera.zoom));
    }

void control_camera_pos(Client* client) {
    // Camera movement with WASD
    float base_camera_speed = 400.0f; // pixels per second at 1x zoom
    float camera_speed = base_camera_speed / client->camera.zoom; // Adjust speed based on zoom
    float delta_time = GetFrameTime();
    
    if (IsKeyDown(KEY_W)) client->camera.target.y -= camera_speed * delta_time;
    if (IsKeyDown(KEY_S)) client->camera.target.y += camera_speed * delta_time;
    if (IsKeyDown(KEY_A)) client->camera.target.x -= camera_speed * delta_time;
    if (IsKeyDown(KEY_D)) client->camera.target.x += camera_speed * delta_time;

    // Calculate world dimensions
    float world_width = client->background_summer.texture.width;
    float world_height = client->background_summer.texture.height;
        
    // Clamp camera to prevent showing empty space
    float half_screen_width = GAME_WIDTH / (2.0f * client->camera.zoom);
    float half_screen_height = GAME_HEIGHT / (2.0f * client->camera.zoom);
    
    client->camera.target.x = fmaxf(half_screen_width, fminf(world_width - half_screen_width, client->camera.target.x));
    client->camera.target.y = fmaxf(half_screen_height, fminf(world_height - half_screen_height, client->camera.target.y));
}

void render_fixed_mode(Client* client, Territories* env) {
    float half_screen_w = GAME_WIDTH / (2.0f * client->camera.zoom);
    float half_screen_h = GAME_HEIGHT / (2.0f * client->camera.zoom);
    float left = client->camera.target.x - half_screen_w;
    float right = client->camera.target.x + half_screen_w;
    float top = client->camera.target.y - half_screen_h;
    float bottom = client->camera.target.y + half_screen_h;

    int start_c = fmaxf(0, floorf(left / TILE_SIZE));
    int end_c = fminf(env->width, ceilf(right / TILE_SIZE));
    int start_r = fmaxf(0, floorf(top / TILE_SIZE));
    int end_r = fminf(env->height, ceilf(bottom / TILE_SIZE));

    float tex_w = client->background_summer.texture.width;   // texture width in pixels
    float tex_h = client->background_summer.texture.height; 

    float src_x = start_c * TILE_SIZE;
    float src_y = start_r * TILE_SIZE;
    float src_w = (end_c - start_c) * TILE_SIZE;
    float src_h = (end_r - start_r) * TILE_SIZE;

    // clamp so we never sample outside the texture
    if (src_x < 0) { src_w += src_x; src_x = 0; }          // adjust width if x negative
    if (src_y < 0) { src_h += src_y; src_y = 0; }
    if (src_x + src_w > tex_w)  src_w = tex_w - src_x;
    if (src_y + src_h > tex_h)  src_h = tex_h - src_y;

    // RenderedTexture2D has a different coordinate system (origin is at the bottom left and y goes upwards).
    // So we need to set the src_y to the bottom of the texture (in its coordinates) and flip the height so that it reads upwards.
    Rectangle src = {src_x, tex_h - src_y - src_h, src_w, -src_h};

    Vector2 dest = {src_x, src_y};
    DrawTextureRec(env->is_winter ? client->background_winter.texture : client->background_summer.texture, src, dest, WHITE);
}

void render_tracking_mode(Client* client, Territories* env) {
    int R = env->height;
    int C = env->width;
    float world_px_w = C * TILE_SIZE;
    float world_px_h = R * TILE_SIZE;

    RenderTexture2D bg = env->is_winter ? client->background_winter : client->background_summer;
    float tex_w = bg.texture.width;
    float tex_h = bg.texture.height;

    // Compute the visible rectangle in world coordinates
    float zoom = client->camera.zoom;
    float view_left = client->camera.target.x - GAME_WIDTH / 2.0f / zoom;
    float view_top  = client->camera.target.y - GAME_HEIGHT / 2.0f / zoom;
    float view_width = GAME_WIDTH / zoom;
    float view_height = GAME_HEIGHT / zoom;

    // For toroidal wrapping, we need to draw up to 4 rectangles
    // Calculate how to split the view rectangle
    
    // X-axis: determine if view wraps horizontally
    float view_right = view_left + view_width;
    float x_wrap_point = -1; // where to split in X (screen coordinates)
    if (view_left < 0) {
        x_wrap_point = -view_left; // distance from left edge to world origin
    } else if (view_right > world_px_w) {
        x_wrap_point = world_px_w - view_left; // distance from left edge to world wrap
    }
    
    // Y-axis: determine if view wraps vertically  
    float view_bottom = view_top + view_height;
    float y_wrap_point = -1; // where to split in Y (screen coordinates)
    if (view_top < 0) {
        y_wrap_point = -view_top; // distance from top edge to world origin
    } else if (view_bottom > world_px_h) {
        y_wrap_point = world_px_h - view_top; // distance from top edge to world wrap
    }
    
    // Draw up to 4 rectangles to cover the view
    for (int x_part = 0; x_part < 2; x_part++) {
        for (int y_part = 0; y_part < 2; y_part++) {
            
            // Skip unnecessary parts
            if (x_part == 1 && x_wrap_point < 0) continue;
            if (y_part == 1 && y_wrap_point < 0) continue;
            
            // Calculate rectangle bounds in view coordinates
            float rect_left, rect_width;
            if (x_wrap_point < 0) {
                // No X wrapping
                rect_left = 0;
                rect_width = view_width;
            } else {
                if (x_part == 0) {
                    rect_left = 0;
                    rect_width = x_wrap_point;
                } else {
                    rect_left = x_wrap_point;
                    rect_width = view_width - x_wrap_point;
                }
            }
            
            float rect_top, rect_height;
            if (y_wrap_point < 0) {
                // No Y wrapping
                rect_top = 0;
                rect_height = view_height;
            } else {
                if (y_part == 0) {
                    rect_top = 0;
                    rect_height = y_wrap_point;
                } else {
                    rect_top = y_wrap_point;
                    rect_height = view_height - y_wrap_point;
                }
            }
            
            if (rect_width <= 0 || rect_height <= 0) continue;
            
            // Convert to world coordinates
            float world_left = view_left + rect_left;
            float world_top = view_top + rect_top;
            
            // Wrap world coordinates to texture space
            float src_x = fmodf(world_left, world_px_w);
            if (src_x < 0) src_x += world_px_w;
            float src_y = fmodf(world_top, world_px_h);
            if (src_y < 0) src_y += world_px_h;
            
            // Ensure we don't exceed texture bounds
            float actual_width = fminf(rect_width, tex_w - src_x);
            float actual_height = fminf(rect_height, tex_h - src_y);
            
            if (actual_width <= 0 || actual_height <= 0) continue;
            
            // RenderTexture2D has flipped Y coordinates
            Rectangle src = {src_x, tex_h - src_y - actual_height, actual_width, -actual_height};
            Vector2 dest = {world_left, world_top};
            
            DrawTextureRec(bg.texture, src, dest, WHITE);
        }
    }
}

int process_tracking_input(Client* client, Territories* env) {
    if (IsKeyPressed(KEY_D)) {
        int initial_pid = (client->tracking_pid + 1) % env->max_agents;
        for (int i = 0; i < env->max_agents; i++) {
            int pid = (i+initial_pid)%env->max_agents;
            if (env->alive_mask[pid]) {
                client->tracking_pid = pid;
                break;
            }
        } 
    } else if (IsKeyPressed(KEY_A)) {
        int initial_pid = (client->tracking_pid - 1 + env->max_agents) % env->max_agents;
        for (int i = 0; i < env->max_agents; i++) {
            int pid = (initial_pid - i + env->max_agents)%env->max_agents;
            if (env->alive_mask[pid]) {
                client->tracking_pid = pid;
                break;
            }
        } 
    }
    else if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        // Get mouse position in screen coordinates
        Vector2 mouse_screen = GetMousePosition();
        // Convert to world coordinates
        Vector2 mouse_world = GetScreenToWorld2D(mouse_screen, client->camera);
        // Convert to tile indices
        int c = (int)(mouse_world.x / TILE_SIZE);
        int r = (int)(mouse_world.y / TILE_SIZE);
        // Check bounds
        if (r >= 0 && r < env->height && c >= 0 && c < env->width) {
            int adr = r * env->width + c;
            int pid = env->pids_2d[adr];
            if (pid != -1) {
                client->tracking_pid = pid;
            }
        }
    }

    // if (env->render_mode == NORMAL) {
    //     if (IsKeyDown(KEY_W)){
    //         return MOVE_UP;
    //     } else if (IsKeyDown(KEY_S)) {
    //         return MOVE_DOWN;
    //     } else if (IsKeyDown(KEY_A)) {
    //         return MOVE_LEFT;
    //     } else if (IsKeyDown(KEY_D)) {
    //         return MOVE_RIGHT;
    //     }
    // } 
    return -1;
}

int process_replay_input(Client* client, Territories* env) {
    if (IsKeyPressed(KEY_SPACE)) {
        client->is_paused = !client->is_paused;
    } else if (IsKeyPressed(KEY_LEFT) && client->is_paused) {
        return -1;
    } else if (IsKeyPressed(KEY_RIGHT) && client->is_paused) {
        return 1;
    }
    return client->is_paused ? 0 : 1;
}

void render_stats_panel(Territories* env) {
    Client* client = env->client;
    
    // Draw background for stats panel
    Rectangle stats_bg = {GAME_WIDTH, 0, SCREEN_WIDTH - GAME_WIDTH, SCREEN_HEIGHT};
    DrawRectangleRec(stats_bg, (Color){40, 40, 40, 255}); // Dark gray background
    
    // Draw border
    DrawRectangleLinesEx(stats_bg, 2, (Color){80, 80, 80, 255});
    
    // Stats text
    int text_x = GAME_WIDTH + 20;
    int text_y = 30;
    int line_height = 25;
    
    // Title
    DrawText("STATISTICS", text_x, text_y, 20, WHITE);
    text_y += 40;
    
    // Current stats (always shown)
    DrawText(TextFormat("Tick: %d", env->tick), text_x, text_y, 16, LIGHTGRAY);
    text_y += line_height;
    
    DrawText(TextFormat("Season: %s", env->is_winter ? "Winter" : "Summer"), text_x, text_y, 16, LIGHTGRAY);
    text_y += line_height;

    int world_pop = 0; // not using agent_manager->alive_count because it's not recorded for replay
    for (int i = 0; i < env->max_agents; i++) {
        if (env->alive_mask[i]) {
            world_pop++;
        }
    }
    
    DrawText(TextFormat("Population: %d", world_pop), text_x, text_y, 16, LIGHTGRAY);
    text_y += line_height;
    
    text_y += 20; // Spacing
    
    if (client->tracking_mode) {
        // Agent tracking stats
        Agent* agent = &env->agents[client->tracking_pid];
        DrawText("Tracked Agent:", text_x, text_y, 18, YELLOW);
        text_y += line_height + 10;
        
        DrawText(TextFormat("PID: %d", client->tracking_pid), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("HP: %d/%d", agent->hp, agent->hp_max), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Satiation: %d/%d", agent->satiation, agent->max_satiation), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Age: %d", agent->age), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Food: %d", agent->food_carried), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Stone: %d", agent->stone_carried), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Role: %d", agent->role), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        // DNA (show first few genes)
        char dna_str[64] = "";
        for (int i = 0; i < min(env->n_genes, 8); i++) {
            char gene_str[8];
            snprintf(gene_str, sizeof(gene_str), "%d", env->dnas[client->tracking_pid * env->n_genes + i]);
            strcat(dna_str, gene_str);
            if (i < min(env->n_genes, 8) - 1) strcat(dna_str, ",");
        }
        if (env->n_genes > 8) strcat(dna_str, "...");
        DrawText(TextFormat("DNA: %s", dna_str), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Reward: %.3f", env->rewards[client->tracking_pid]), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        // Action names
        const char* action_names[] = {
            "MOVE_UP", "MOVE_RIGHT", "MOVE_DOWN", "MOVE_LEFT", "NOOP",
            "PICKUP", "MINE", "PACKAGE_FOOD", "BUILD_WALL", "ATTACK", "REPRODUCE"
        };
        int action = env->actions[client->tracking_pid];
        DrawText(TextFormat("Action: %s", action_names[action]), text_x, text_y, 16, LIGHTGRAY);
       
    } else {
        // Episode stats
        DrawText("Episode Stats:", text_x, text_y, 18, YELLOW);
        text_y += line_height + 10;
        
        DrawText(TextFormat("Births: %.0f", env->stats.births), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Starvations: %.0f", env->stats.starvations), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Murders: %.0f", env->stats.murders), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Stone Mined: %.0f", env->stats.stone_mined), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Walls Built: %.0f", env->stats.walls_built), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Walls Destroyed: %.0f", env->stats.wall_destroyed), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Food Stored: %.0f", env->stats.food_stored), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Food Eaten: %.0f", env->stats.food_eaten), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        text_y += 20; // Spacing
        
        DrawText("Population Stats:", text_x, text_y, 18, YELLOW);
        text_y += line_height + 10;
        
        DrawText(TextFormat("Max Population: %.0f", env->stats.max_pop), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Min Population: %.0f", env->stats.min_pop), text_x, text_y, 16, LIGHTGRAY);
        text_y += line_height;
        
        DrawText(TextFormat("Avg Population: %.1f", env->stats.avg_population), text_x, text_y, 16, LIGHTGRAY);
    }
}

// Required function. Should handle creating the client on first call
int c_render(Territories* env) {
    /*
    If in replay mode:
    - Output is the next tick to render

    Else:
    - Output is either -1 or the action for the agent being controlled to take

    */
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* client = env->client;
    // Handle Inputs
    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }
    if (IsKeyPressed(KEY_T)) {
        client->tracking_mode = !client->tracking_mode;
    }
    if (client->tracking_mode) {
        bool found = false;
        for (int i = 0; i < env->max_agents; i++) {
            int pid = (i+client->tracking_pid)%env->max_agents;
            if (env->alive_mask[pid]) {
                client->tracking_pid = pid;
                found = true;
                break;
            }
        } 
        if (!found) {
            client->tracking_mode = false;
        } else {
            Agent* agent = &env->agents[client->tracking_pid];
            client->camera.target.x = (agent->c + 0.5f) * TILE_SIZE;
            client->camera.target.y = (agent->r + 0.5f) * TILE_SIZE;
        }
    }

    BeginDrawing();
    ClearBackground(BLANK);
    BeginMode2D(client->camera);
    control_camera_zoom(client);
    int output = -1;
    if (env->render_mode == REPLAY) {
        output = process_replay_input(client, env);
    }
    if (client->tracking_mode) {
        int action = process_tracking_input(client, env);
        if (env->render_mode == NORMAL) {
            output = action;
        }
        render_tracking_mode(client, env);
    }
    else{
        render_fixed_mode(client, env);
        control_camera_pos(client);
    }
    // render agents and resources
    float half_screen_w = GAME_WIDTH / (2.0f * client->camera.zoom);
    float half_screen_h = GAME_HEIGHT / (2.0f * client->camera.zoom);
    float left = client->camera.target.x - half_screen_w;
    float right = client->camera.target.x + half_screen_w;
    float top = client->camera.target.y - half_screen_h;
    float bottom = client->camera.target.y + half_screen_h;

    int start_c = floorf(left / TILE_SIZE);
    int end_c = ceilf(right / TILE_SIZE);
    int start_r = floorf(top / TILE_SIZE);
    int end_r = ceilf(bottom / TILE_SIZE);
    if (!client->tracking_mode) {
        start_c = fmaxf(0, start_c);
        end_c = fminf(env->width, end_c);
        start_r = fmaxf(0, start_r);
        end_r = fminf(env->height, end_r);
    }
    int R = env->height;
    int C = env->width;
    for (int rr = start_r; rr < end_r; rr++) {
        for (int cc = start_c; cc < end_c; cc++) {
            int r = (rr + R) % R;
            int c = (cc + C) % C;
            int adr = r*env->width + c;
            Vector2 dest_pos = {cc*TILE_SIZE, rr*TILE_SIZE};
            // draw resources and walls
            if (env->tile_props[adr*4 + STORED_FOOD] > 0) {
                int u = TILE_SIZE * 1;
                DrawTextureRec(client->food_sprite, (Rectangle){u, 0, TILE_SIZE, TILE_SIZE}, dest_pos, WHITE);
            } else if (env->tile_props[adr*4 + STONE] > 0) {
                int u = (24 + adr % 2) * TILE_SIZE;
                int v = env->is_winter ? TILE_SIZE : 0;
                DrawTextureRec(client->terrain_sprite, (Rectangle){u, v, TILE_SIZE, TILE_SIZE}, dest_pos, WHITE);
            }
            else if (env->tile_props[adr*4 + WALL_HP] > 0) {
                int u = TILE_SIZE * client->wall_sprite_indices[adr];
                DrawTextureRec(client->wall_sprite, (Rectangle){u, 0, TILE_SIZE, TILE_SIZE}, 
                (Vector2){c*TILE_SIZE, r*TILE_SIZE}, WHITE);
            }
            else {
                int growth_days = get_growth_days(env, r, c);
                if (growth_days > 0) {
                    int crop_available = (int)(exp(K * growth_days) - 1);
                    int crop_stage = (int)round(((float)crop_available / client->max_crop_available) * 5); // 5 stages of crop growth (1-5) + 1 empty stage (nothing there)
                    if (crop_stage > 0) {
                        int u = TILE_SIZE*(crop_stage + 2);
                        DrawTextureRec(client->food_sprite, (Rectangle){u, 0, TILE_SIZE, TILE_SIZE}, dest_pos, WHITE);
                    }
                } 
            }
                
            // draw agents
            int pid = env->pids_2d[adr];
            if (pid != -1) {
                Agent* agent = &env->agents[pid];
                int sprite_r = agent->dir;
                Vector2 pos = {(cc - 0.5f)*TILE_SIZE, (rr - 0.5f)*TILE_SIZE};

                for (int i = 0; i < env->n_genes; i++) {
                    int allele = env->dnas[pid*env->n_genes + i];
                    Texture2D text = client->char_bases[i*env->n_alleles + allele];
                    if (text.id != 0) {
                        DrawTextureRec(text, 
                            (Rectangle){0, sprite_r*SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE}, 
                            pos, WHITE);
                    }
                }
            }

        }
    }

    if (env->render_mode == REPLAY) {
        env->tick += output;
    }

    EndMode2D();
    
    // Render stats panel on the right side
    render_stats_panel(env);
    
    EndDrawing();
    return output;
}



// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals, alive, kinship_matrix, dnas
// Those vectors were allocated by Python and will be freed by Python's garbage collector
void c_close(Territories* env) {
    if (env->agent_manager != NULL) {
        bitset_free(env->agent_manager->alive_bitset);
        free(env->agent_manager->free_pids);
        free(env->agent_manager->alive_pids);
        free(env->agent_manager);
    }
    free(env->agents);
    free(env->pids_2d);
    free(env->is_soil);
    free(env->tile_props);
    free(env->family_sizes);
    free(env->prev_family_sizes);
    if (env->client != NULL) {
        Client* client = env->client;
        UnloadTexture(client->terrain_sprite);
        UnloadRenderTexture(client->background_summer);
        UnloadRenderTexture(client->background_winter);
        UnloadTexture(client->wall_sprite);
        UnloadTexture(client->food_sprite);
        for (int i = 0; i < env->n_alleles * env->n_genes; i++) {
            UnloadTexture(client->char_bases[i]);
        }
        free(client->terrain_sprite_indices);
        free(client->wall_sprite_indices);
        free(client->char_bases);
        free(client);
    }
}