using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[System.Serializable]
public class MotionFrame
{
    public int step;
    public float[] root_pos;
    public float root_angle;
    public float[] joints;
}

[System.Serializable]
public class MotionData
{
    public List<MotionFrame> frames;
    public MotionMetadata metadata;
}

[System.Serializable]
public class MotionMetadata
{
    public string env_id;
    public string algo;
    public float dt;
}

public class MotionImporter : MonoBehaviour
{
    [Header("Data Source")]
    public string jsonFilePath = "motion_data.json";
    
    [Header("Rig References")]
    public Transform rootBone;  // The main body/torso
    public Transform thighBone; // First leg segment
    public Transform legBone;   // Second leg segment
    public Transform footBone;  // Foot segment

    [Header("Playback Settings")]
    public float speedMultiplier = 1.0f;
    public bool loop = true;

    private MotionData motionData;
    private float currentTime = 0f;
    private float totalDuration = 0f;
    private bool isLoaded = false;

    void Start()
    {
        LoadMotionData();
    }

    void LoadMotionData()
    {
        string fullPath = Path.Combine(Application.dataPath, jsonFilePath);
        if (!File.Exists(fullPath))
        {
            // Try relative to project root if not in Assets
            fullPath = jsonFilePath; 
        }

        if (File.Exists(fullPath))
        {
            string jsonContent = File.ReadAllText(fullPath);
            motionData = JsonUtility.FromJson<MotionData>(jsonContent);
            
            if (motionData != null && motionData.frames != null && motionData.frames.Count > 0)
            {
                // Calculate duration based on frame count and dt (defaulting to 0.008s if missing)
                float dt = motionData.metadata != null && motionData.metadata.dt > 0 ? motionData.metadata.dt : 0.008f;
                totalDuration = motionData.frames.Count * dt;
                isLoaded = true;
                Debug.Log($"Loaded {motionData.frames.Count} frames. Duration: {totalDuration}s");
            }
            else
            {
                Debug.LogError("Failed to parse motion data or data is empty.");
            }
        }
        else
        {
            Debug.LogError($"Motion data file not found at: {fullPath}");
        }
    }

    void Update()
    {
        if (!isLoaded) return;

        // Advance time
        currentTime += Time.deltaTime * speedMultiplier;
        
        if (currentTime > totalDuration)
        {
            if (loop)
                currentTime %= totalDuration;
            else
                currentTime = totalDuration;
        }

        // Find current frame index
        float dt = motionData.metadata != null && motionData.metadata.dt > 0 ? motionData.metadata.dt : 0.008f;
        int frameIndex = Mathf.FloorToInt(currentTime / dt);
        frameIndex = Mathf.Clamp(frameIndex, 0, motionData.frames.Count - 1);

        ApplyPose(motionData.frames[frameIndex]);
    }

    void ApplyPose(MotionFrame frame)
    {
        // 1. Root Position
        // MuJoCo: X (forward), Y (up/angle?), Z (up in 3D, but Hopper is 2D planar)
        // Wait, MuJoCo standard is Z-up. Hopper moves in X (forward) and Z (up). Y is lateral (locked).
        // Unity standard is Y-up. Z is forward.
        // Mapping:
        // MuJoCo X  -> Unity Z (Forward)
        // MuJoCo Z  -> Unity Y (Up)
        // MuJoCo Y  -> Unity X (Lateral - usually 0 for Hopper)
        
        float mj_x = frame.root_pos[0];
        float mj_z = frame.root_pos[1]; // In our python script, we saved qpos[1] as z (height)
        
        Vector3 newPos = new Vector3(0, mj_z, mj_x); 
        rootBone.position = newPos;

        // 2. Root Rotation
        // MuJoCo qpos[2] is angle around Y-axis (which points out of plane in 2D profile).
        // In Unity, if we map X->Z and Z->Y, the 'out of plane' axis is X.
        // Only if we faced the camera sideways.
        // Let's assume standard alignment:
        // Unity rotation around X axis (pitch) corresponds to the main tipping motion of the hopper?
        // Actually, Hopper rotates around the Y-axis in MuJoCo (pitch).
        // In Unity (Z-forward, Y-up), pitch is X-axis rotation.
        
        float rootAngleDegrees = frame.root_angle * Mathf.Rad2Deg;
        // Applying rotation. Depending on Rig, might need offset or different axis.
        // Assuming Identity quaternion defaults to upright.
        rootBone.rotation = Quaternion.Euler(-rootAngleDegrees, 0, 0); // Negative might be needed for coordinate handedness swap

        // 3. Joints
        // frame.joints: [thigh, leg, foot]
        // These are relative rotations (hinges).
        // We need to apply them to local rotations.
        
        if (thighBone)
        {
            float thighAngle = frame.joints[0] * Mathf.Rad2Deg;
            // Assuming the joint rotates around Unity's X-axis (pitch)
            thighBone.localRotation = Quaternion.Euler(0, 0, -thighAngle); // Axis depends on rig import. Often Z for 2D sprites or X for 3D rigs.
            // Trial and error often needed here without seeing the rig.
        }

        if (legBone)
        {
            float legAngle = frame.joints[1] * Mathf.Rad2Deg;
            legBone.localRotation = Quaternion.Euler(0, 0, -legAngle);
        }

        if (footBone)
        {
            float footAngle = frame.joints[2] * Mathf.Rad2Deg;
            footBone.localRotation = Quaternion.Euler(0, 0, -footAngle);
        }
    }
}
