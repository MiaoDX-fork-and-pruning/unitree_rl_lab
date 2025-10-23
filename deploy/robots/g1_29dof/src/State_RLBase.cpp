#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// Persistent WASD velocity control: updates only the relevant axis and persists until changed.
// Space resets all components to zero.
// Clamps to ranges from deploy.yaml (commands.base_velocity.ranges).
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    static std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    std::string key = FSMState::keyboard->key();
    // normalize to lowercase single-char key if applicable
    if (key.size() == 1) {
        key[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(key[0])));
    }

    const auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];
    const float min_x = ranges["lin_vel_x"][0].as<float>();
    const float max_x = ranges["lin_vel_x"][1].as<float>();
    const float min_y = ranges["lin_vel_y"][0].as<float>();
    const float max_y = ranges["lin_vel_y"][1].as<float>();
    const float min_z = ranges["ang_vel_z"][0].as<float>();
    const float max_z = ranges["ang_vel_z"][1].as<float>();

    // desired constants, capped by configured maxima
    const float vx = std::min(0.5f, max_x);
    const float vy = std::min(0.3f, max_y);
    const float wz = std::min(0.2f, max_z);

    // Persist previous command; only change on recognized keys. This avoids cross-thread race on on_pressed.
    static std::vector<float> prev = {0.0f, 0.0f, 0.0f};
    bool updated = false;
    if (key == "w") { // forward only
        cmd[0] = std::clamp(+vx, min_x, max_x);
        cmd[1] = 0.0f; cmd[2] = 0.0f; updated = true;
    }
    else if (key == "s") { // backward only
        cmd[0] = std::clamp(-vx, min_x, max_x);
        cmd[1] = 0.0f; cmd[2] = 0.0f; updated = true;
    }
    else if (key == "a") { // strafe left only
        cmd[1] = std::clamp(+vy, min_y, max_y);
        cmd[0] = 0.0f; cmd[2] = 0.0f; updated = true;
    }
    else if (key == "d") { // strafe right only
        cmd[1] = std::clamp(-vy, min_y, max_y);
        cmd[0] = 0.0f; cmd[2] = 0.0f; updated = true;
    }
    else if (key == "q") { // yaw left only
        cmd[2] = std::clamp(+wz, min_z, max_z);
        cmd[0] = 0.0f; cmd[1] = 0.0f; updated = true;
    }
    else if (key == "e") { // yaw right only
        cmd[2] = std::clamp(-wz, min_z, max_z);
        cmd[0] = 0.0f; cmd[1] = 0.0f; updated = true;
    }
    else if (key == " ") { // stop
        cmd = {0.0f, 0.0f, 0.0f}; updated = true;
    }
    if (updated && (cmd[0]!=prev[0] || cmd[1]!=prev[1] || cmd[2]!=prev[2])) {
        spdlog::debug("[WASD] cmd vx:{:.2f} vy:{:.2f} wz:{:.2f}", cmd[0], cmd[1], cmd[2]);
        prev = cmd;
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    spdlog::info("Initializing State_{}...", state_string);
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    spdlog::info("[Deploy] Using deploy cfg: {}", (policy_dir / "params" / "deploy.yaml").string());
    spdlog::info("[Deploy] Using ONNX model: {}", (policy_dir / "exported" / "policy.onnx").string());
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            (int)FSMMode::Passive
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
