/** Definitive explainer — used by WhatIs page + static prerender. */

export const WHAT_IS_PATH = '/what-is-ai-badminton-stroke-analysis'

export const WHAT_IS = {
    path: WHAT_IS_PATH,
    seoTitle: 'What Is AI Badminton Stroke Analysis? · IsoCourt',
    seoDescription:
        'AI badminton stroke analysis explains how computer vision reads your rallies: pose tracing, stroke labels, scores, and coaching tips. How IsoCourt does it, and what it cannot do.',
    heroTitleText: 'what is AI badminton stroke analysis?',
    lead: 'Software that watches your badminton clip, labels what you hit, and points at form issues you would otherwise scrub past.',
    updated: 'July 23, 2026',
    sections: [
        {
            id: 'definition',
            h: 'Plain definition',
            p: [
                'AI badminton stroke analysis is video analysis aimed at shots, not full match stats. A model looks at frames of you hitting. It estimates body pose, groups frames into stroke windows, assigns a stroke type (smash, clear, drop, and similar), and often attaches a quality score plus short coaching notes.',
                'IsoCourt does that in the browser: upload a clip or go live with your camera. Birdzo is the coach persona that turns those reads into tips.',
            ],
        },
        {
            id: 'how-it-works',
            h: 'How it works (IsoCourt)',
            p: [
                '1. You bring a rally, drill, or single stroke. Keep the shuttle in frame.',
                '2. The pipeline traces pose (skeleton on your body) and finds stroke windows on a timeline.',
                '3. Each window gets labels and a score. Birdzo writes short cues tied to those moments.',
                '4. You scrub the timeline, check skeleton frames, and take one or two cues back on court.',
            ],
        },
        {
            id: 'you-get',
            h: 'What you get',
            p: [
                'Pose tracing so late split-steps and contact behind the body are visible.',
                'Stroke reads across common shot types, not a generic "good/bad" blob.',
                'A quality score as model judgment for that window, not a ranking of your life.',
                'Tips meant for the next rep. Training insight only.',
            ],
        },
        {
            id: 'limits',
            h: 'What it is not',
            p: [
                'Not a human coach. Not medical advice. Not a tournament umpire.',
                'Not magic on messy doubles footage with two players overlapping and a tiny shuttle.',
                'Not a full course curriculum. If you want drill playlists and levels, a course product fits better. See Compare.',
            ],
        },
        {
            id: 'when',
            h: 'When it helps',
            p: [
                'You already film yourself and waste time guessing what looked off.',
                'You want a fast read between club sessions.',
                'You want live camera feedback without installing another sports app first.',
            ],
        },
    ],
}
