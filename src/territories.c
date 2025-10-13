#include "territories.h"
#include "raylib.h"

int main() {
  Territories env = {.n_genes = 1};
  env.width = 128;
  env.height = 128;
  env.max_agents = 512; // Set max_agents before using it

  // Allocate arrays for standalone C program
  env.n_roles = 1;
  env.min_ep_length = 512;
  env.max_ep_length = 576;

  env.observations = (unsigned char *)calloc(
      env.max_agents * (9 * 9 * (9 + env.n_genes) + 3 + 5),
      sizeof(unsigned char));
  env.actions = (int *)calloc(env.max_agents, sizeof(int));
  env.rewards = (float *)calloc(env.max_agents, sizeof(float));
  env.terminals =
      (unsigned char *)calloc(env.max_agents, sizeof(unsigned char));
  env.alive = (unsigned char *)calloc(env.max_agents, sizeof(unsigned char));
  env.dnas = (unsigned char *)calloc(env.max_agents * env.n_genes,
                                     sizeof(unsigned char));

  init(&env);
  c_reset(&env);
  int action = c_render(&env);
  while (!WindowShouldClose()) {
    for (int i = 0; i < env.max_agents; i++) {
      env.actions[i] = rand() % 11;
    }
    if (env.client->tracking_mode) {
      env.actions[env.client->tracking_pid] = action;
    }

    c_step(&env);
    for (int i = 0; i < 20; i++) {
      action = c_render(&env);
    }
  }

  c_close(&env);
  return 0;
}